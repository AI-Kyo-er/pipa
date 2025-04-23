import pandas as pd
from pandarallel import pandarallel
from pipa.common.hardware.cpu import NUM_CORES_PHYSICAL
from pipa.common.logger import logger
from pipa.common.utils import generate_unique_rgb_color
from pipa.parser import make_single_plot
from typing import List, Optional
import seaborn as sns
import plotly.graph_objects as go
import re


class PerfStatData:
    def __init__(self, perf_stat_csv_path: str):
        self.data = self.parse_perf_stat_file(perf_stat_csv_path)
        self._df_wider = None
        self.use_ef_cores = False
        
    def _extract_core_event(self, event_name: str) -> str:
        """
        Extract the core event from a cpu_core/event/ or cpu_atom/event/ format.
        
        Args:
            event_name (str): The event name to process.
            
        Returns:
            str: The extracted event name or the original if no match.
        """
        if not self.use_ef_cores:
            return event_name
            
        # Use regex to extract the event part from cpu_xxx/event/ format
        match = re.match(r'cpu_\w+/([^/]+)/', event_name)
        if match:
            return match.group(1)
        return event_name

    def get_CPI(self):
        """
        Returns the CPI (Cycles Per Instruction) data.

        Returns:
            pd.DataFrame: Dataframe containing the CPI data.
        """
        # Define event names based on whether use_ef_cores is enabled
        cycles_event = "cycles"
        instructions_event = "instructions"
        
        if self.use_ef_cores:
            # Get all metric types and filter for the ones that match cycles or instructions
            all_metrics = self.data["metric_type"].unique()
            cycles_metrics = [m for m in all_metrics if self._extract_core_event(m) == "cycles"]
            instructions_metrics = [m for m in all_metrics if self._extract_core_event(m) == "instructions"]
            
            # Get cycles data from all matching events and drop rows with NaN values
            cycles_df = self.data[self.data["metric_type"].isin(cycles_metrics)].dropna(subset=["value"])
            # Add a new column with the extracted event name
            cycles_df["core_event"] = cycles_df["metric_type"].apply(self._extract_core_event)
            
            # Get instructions data from all matching events and drop rows with NaN values
            instructions_df = self.data[self.data["metric_type"].isin(instructions_metrics)].dropna(subset=["value"])
            # Add a new column with the extracted event name
            instructions_df["core_event"] = instructions_df["metric_type"].apply(self._extract_core_event)
        else:
            # Get cycles data and drop rows with NaN values
            cycles_df = self.data[self.data["metric_type"] == cycles_event].dropna(subset=["value"])
            cycles_df["core_event"] = cycles_event
            
            # Get instructions data and drop rows with NaN values
            instructions_df = self.data[self.data["metric_type"] == instructions_event].dropna(subset=["value"])
            instructions_df["core_event"] = instructions_event
        
        # Merge the dataframes
        merged_df = cycles_df.merge(
            instructions_df,
            on=["timestamp", "cpu_id", "core_event"],
            suffixes=("_cycles", "_instructions"),
            how="inner"  # Only keep rows where both cycles and instructions are available
        )
        
        # Calculate CPI only for valid rows (avoid division by zero)
        valid_mask = merged_df["value_instructions"] > 0
        merged_df["CPI"] = float('nan')  # Initialize with NaN
        merged_df.loc[valid_mask, "CPI"] = merged_df.loc[valid_mask, "value_cycles"] / merged_df.loc[valid_mask, "value_instructions"]
        
        return merged_df.drop(columns=["metric_type_cycles", "metric_type_instructions"])

    def get_CPI_time(self, threads: list | None = None):
        """
        Returns the CPI (Cycles Per Instruction) over time for the specified threads.

        Args:
            threads (list, optional): A list of thread IDs. If None, returns the average CPI over time for all threads.

        Returns:
            pandas.DataFrame: A DataFrame containing the timestamp and CPI values over time.

        """
        if threads is None:
            return self.get_CPI()[["timestamp", "CPI"]].groupby("timestamp").mean()
        df = self.get_CPI()
        return df[df["cpu_id"].isin([int(t) for t in threads])]

    def get_CPI_overall(self, data_type="thread"):
        """
        Calculate the overall CPI (Cycles Per Instruction) based on the given data type.

        Parameters:
        - data_type (str): The type of data to calculate CPI for. Can be "thread" or "system".

        Returns:
        - If data_type is "thread", returns a DataFrame with CPI values per thread.
        - If data_type is "system", returns the overall CPI value for the system.

        Raises:
        - ValueError: If an invalid data type is provided.
        """

        df = self.get_CPI()
        # Drop rows with NaN values in either cycles or instructions
        df = df.dropna(subset=["value_cycles", "value_instructions"])
        
        match data_type:
            case "thread":
                data_per_thread = (
                    df[["cpu_id", "value_cycles", "value_instructions"]]
                    .groupby("cpu_id")
                    .sum()
                )
                # Avoid division by zero or NaN
                valid_mask = (data_per_thread["value_instructions"] > 0) & (data_per_thread["value_cycles"].notna())
                data_per_thread["CPI"] = float('nan')  # Initialize with NaN
                data_per_thread.loc[valid_mask, "CPI"] = (
                    data_per_thread.loc[valid_mask, "value_cycles"]
                    / data_per_thread.loc[valid_mask, "value_instructions"]
                )
                return data_per_thread
            case "system":
                total_instructions = df["value_instructions"].sum()
                total_cycles = df["value_cycles"].sum()
                if total_instructions > 0:
                    return total_cycles / total_instructions
                else:
                    logger.warning("Total instructions is zero or NaN, cannot calculate system CPI")
                    return float('nan')
            case _:
                raise ValueError("Invalid data type")

    def get_CPI_by_thread(self, threads: list):
        """
        Returns the weighted CPI (Cycles Per Instruction) for the specified threads.

        Args:
            threads (list): A list of thread IDs.

        Returns:
            float: The weighted CPI value for the specified threads.
        """
        df = self.get_CPI_overall("thread")
        # Check if all threads exist in the dataframe
        avail_threads = [t for t in threads if t in df.index]
        if len(avail_threads) < len(threads):
            missing = set(threads) - set(avail_threads)
            logger.warning(f"Some threads {missing} are not found in the data for CPI calculation")
        
        # Use only available threads with valid data
        df_subset = df.loc[avail_threads]
        cycles_sum = df_subset["value_cycles"].dropna().sum()
        instructions_sum = df_subset["value_instructions"].dropna().sum()
        
        if instructions_sum > 0:
            return cycles_sum / instructions_sum
        else:
            logger.warning("Total instructions for the specified threads is zero or NaN, cannot calculate CPI")
            return float('nan')

    def get_CPI_average_by_thread(self, threads: list):
        """
        Returns the average CPI (Cycles Per Instruction) for the specified threads.

        Args:
            threads (list): A list of thread IDs.

        Returns:
            float: The average CPI value for the specified threads.
        """
        return self.get_CPI_overall("thread").loc[threads]["CPI"].mean()

    def plot_CPI_time_by_thread(self, threads: list):
        """
        Plots CPI over time for the specified threads.

        Args:
            threads (list): A list of thread IDs.

        Returns:
            None
        """
        sns.set_theme(style="darkgrid", rc={"figure.figsize": (15, 8)})
        if len(threads) > 1:
            p = sns.lineplot(
                data=self.get_CPI_time(threads), x="timestamp", hue="cpu_id", y="CPI"
            )
        else:
            p = sns.lineplot(data=self.get_CPI_time(threads), x="timestamp", y="CPI")
        p.set_title("CPI over Time, Thread " + ",".join([str(t) for t in threads]))
        p.set_xlabel("Time(s)")
        p.set_ylabel("CPI")
        return p

    def plot_CPI_time_system(self):
        """
        Plots CPI (Cycles Per Instruction) over time for the system.

        This method generates a line plot showing the CPI values over time for the system.
        It uses the data returned by the `get_CPI_time` method and saves the plot as an image.

        Returns:
            None
        """
        sns.set_theme(style="darkgrid", rc={"figure.figsize": (15, 8)})
        p = sns.lineplot(data=self.get_CPI_time(), x="timestamp", y="CPI")
        p.set_title("CPI over Time, System")
        p.set_xlabel("Time(s)")
        p.set_ylabel("CPI")
        return p

    def get_events_overall(self, events: str, data_type="thread"):
        """
        Calculate the overall events based on the given data type.

        Parameters:
        - events (str): The type of events to calculate. Can be "cache-references", "cache-misses", "branch-misses", etc.
        - data_type (str): The type of data to calculate events for. Can be "thread" or "system".

        Returns:
        - If data_type is "thread", returns a DataFrame with events values per thread.
        - If data_type is "system", returns the overall events value for the system.

        Raises:
        - ValueError: If an invalid data type is provided.
        """
        # If using ef_cores, need to match event names differently
        if self.use_ef_cores:
            # Get all metric types
            all_metrics = self.data["metric_type"].unique()
            # Filter metrics that match the desired event after extraction
            matching_metrics = [m for m in all_metrics if self._extract_core_event(m) == events]
            if not matching_metrics:
                logger.warning(f"No metrics found matching '{events}' after extraction")
                # Return empty dataframe or 0 based on data_type
                if data_type == "thread":
                    return pd.DataFrame(columns=["cpu_id", "value"]).set_index("cpu_id")
                else:
                    return 0
            
            # Get data for all matching metrics
            df = self.data[self.data["metric_type"].isin(matching_metrics)]
        else:
            df = self.data[self.data["metric_type"] == events]
            
        match data_type:
            case "thread":
                return df[["cpu_id", "value"]].groupby("cpu_id").sum()
            case "system":
                return df["value"].sum()
            case _:
                raise ValueError("Invalid data type")

    def get_cycles_overall(self, data_type="thread"):
        """
        Calculate the overall cycles based on the given data type.

        Parameters:
        - data_type (str): The type of data to calculate cycles for. Can be "thread" or "system".

        Returns:
        - If data_type is "thread", returns a DataFrame with cycles values per thread.
        - If data_type is "system", returns the overall cycles value for the system.

        Raises:
        - ValueError: If an invalid data type is provided.
        """
        return self.get_events_overall("cycles", data_type)

    def get_instructions_overall(self, data_type="thread"):
        """
        Calculate the overall instructions based on the given data type.

        Parameters:
        - data_type (str): The type of data to calculate instructions for. Can be "thread" or "system".

        Returns:
        - If data_type is "thread", returns a DataFrame with instructions values per thread.
        - If data_type is "system", returns the overall instructions value for the system.

        Raises:
        - ValueError: If an invalid data type is provided.
        """
        return self.get_events_overall("instructions", data_type)

    def get_cycles_by_thread(self, threads=None):
        """
        Returns the total cycles per thread.

        Args:
            threads (list): A list of thread IDs.

        Returns:
            pd.DataFrame: A DataFrame containing the total cycles per thread.
        """
        if threads is None:
            # Use dropna to ignore NaN values
            return self.get_cycles_overall("thread")["value"].dropna().sum()
        
        df = self.get_cycles_overall("thread")
        # Check if all threads exist in the dataframe
        avail_threads = [t for t in threads if t in df.index]
        if len(avail_threads) < len(threads):
            missing = set(threads) - set(avail_threads)
            logger.warning(f"Some threads {missing} are not found in the data")
        
        # Use dropna to ignore NaN values
        return df.loc[avail_threads]["value"].dropna().sum()

    def get_instructions_by_thread(self, threads=None):
        """
        Returns the total instructions per thread.

        Args:
            threads (list): A list of thread IDs.

        Returns:
            int: The total instructions in all threads used.
        """
        if threads is None:
            # Use dropna to ignore NaN values for system-wide calculation
            if self.use_ef_cores:
                # If using ef_cores, need to match event names differently
                all_metrics = self.data["metric_type"].unique()
                # Filter metrics that match instructions after extraction
                matching_metrics = [m for m in all_metrics if self._extract_core_event(m) == "instructions"]
                if not matching_metrics:
                    logger.warning("No metrics found matching 'instructions' after extraction")
                    return float('nan')
                
                # Get data for all matching metrics
                df = self.data[self.data["metric_type"].isin(matching_metrics)]
            else:
                df = self.data[self.data["metric_type"] == "instructions"]
            
            return df["value"].dropna().sum()
            
        # For specific threads
        df = self.get_instructions_overall("thread")
        # Check if all threads exist in the dataframe
        avail_threads = [t for t in threads if t in df.index]
        if len(avail_threads) < len(threads):
            missing = set(threads) - set(avail_threads)
            logger.warning(f"Some threads {missing} are not found in the data")
            
        # Use dropna to ignore NaN values
        return df.loc[avail_threads]["value"].dropna().sum()

    def get_pathlength(self, num_transcations: int, threads: list):
        """
        Returns the pathlength for the given number of transcations and threads.

        Args:
            num_transcations (int): The number of transcations.
            threads (list): A list of thread IDs.

        Returns:
            float: The pathlength value.
        """
        insns = self.get_instructions_by_thread(threads)
        path_length = insns / num_transcations
        return path_length

    def get_cycles_per_second(self, seconds: int = 120, threads=None):
        """
        Returns the cycles per second.

        Args:
            seconds (int): The number of seconds.
            threads (list): A list of thread IDs.

        Returns:
            float: The cycles per second value.
        """
        cycles = self.get_cycles_by_thread(threads)
        # Check if cycles is valid (not NaN)
        if pd.isna(cycles):
            logger.warning("Cycles value is NaN, cannot calculate cycles per second")
            return float('nan')
        return cycles / seconds

    def get_instructions_per_second(self, seconds: int = 120, threads=None):
        """
        Returns the instructions per second.

        Args:
            seconds (int): The number of seconds.
            threads (list): A list of thread IDs.

        Returns:
            float: The instructions per second value.
        """
        instructions = self.get_instructions_by_thread(threads)
        # Check if instructions is valid (not NaN)
        if pd.isna(instructions):
            logger.warning("Instructions value is NaN, cannot calculate instructions per second")
            return float('nan')
        return instructions / seconds

    def get_time_range(self):
        """
        Returns the time range of the data.

        Returns:
            tuple: A tuple containing the minimum and maximum timestamps.
        """
        return self.data["timestamp"].min(), self.data["timestamp"].max()

    def get_time_delta(self):
        """
        Returns the time delta of the data.

        Returns:
            float: The time delta between timestamps.
        """
        return self.data["timestamp"].diff().mean()

    def get_time_total(self):
        """
        Returns the total time of the data.

        Returns:
            float: The total time of the data.
        """
        return self.data["timestamp"].max() - self.data["timestamp"].min()  # in seconds

    def is_multiplexing(self):
        """
        Check if the data contains multiplexing.

        Returns:
            bool: True if the data contains multiplexing, False otherwise.
        """
        return all(self.data["run_percentage"].astype(int) == 100)

    def get_wider_data(self):
        """
        Get the wider data with columns for each metric type.
        Tidy the data by pivoting the metric_type column.

        Returns:
            pd.DataFrame: The wider data.
        """
        if self._df_wider is not None:
            return self._df_wider
            
        df = self.data[["timestamp", "cpu_id", "value", "metric_type"]]
        
        # Apply event name extraction for ef_cores mode
        if self.use_ef_cores:
            # Create a new column with extracted event names for pivot
            df["core_event"] = df["metric_type"].apply(self._extract_core_event)
            df_wider = df.pivot_table(
                index=["timestamp", "cpu_id"],
                columns="core_event",
                values="value",
                aggfunc="first",
            ).reset_index()
        else:
            df_wider = df.pivot_table(
                index=["timestamp", "cpu_id"],
                columns="metric_type",
                values="value",
                aggfunc="first",
            ).reset_index()
            
        df_wider.columns = [f"{col}" for col in df_wider.columns]
        self._df_wider = df_wider
        return df_wider

    def get_tidy_data(self, thread_list: list = None):
        """
        Get the tidied data with columns for each metric type.
        Tidy the data by pivoting the metric_type column.

        ```
        Args:
            thread_list (list, optional): A list of hardware thread names to include in the tidy data.
            If None, all threads are included. Default is None.

        Returns:
            pd.DataFrame: The tidied data.
        ```
        """
        df = self.get_wider_data()
        if thread_list:
            thread_list = [int(cpu) for cpu in thread_list]
            df = df[df["cpu_id"].isin(thread_list)]
            if len(thread_list) == 1:
                return df
        df_t = df.pivot_table(index=["timestamp"], columns="cpu_id").reset_index()
        df_t.columns = [f"{col[0]}_{col[1]}" for col in df_t.columns]
        df_t.rename(columns={"timestamp_": "timestamp"}, inplace=True)
        return df_t

    @staticmethod
    def parse_perf_stat_file(stat_output_path: str):
        """
        Parse the perf stat output file and return a pandas DataFrame.

        Args:
            stat_output_path (str): The path to the perf stat output file.

        Returns:
            pandas.DataFrame: The parsed data as a DataFrame.

        The fields are in this order:
        -   optional usec time stamp in fractions of second (with -I xxx)
        -   optional CPU, core, or socket identifier
        -   optional number of logical CPUs aggregated
        -   counter value
        -   unit of the counter value or empty
        -   event name
        -   run time of counter
        -   percentage of measurement time the counter was running
        -   optional metric value
        -   optional unit of metric
        """
        pandarallel.initialize(min(12, NUM_CORES_PHYSICAL))
        
        # First read the CSV with value as string to handle '<not supported>'
        df = pd.read_csv(
            stat_output_path,
            skiprows=1,
            names=[
                "timestamp",
                "cpu_id",
                "value",
                "unit",
                "metric_type",
                "run_time(ns)",
                "run_percentage",
                "opt_value",
                "opt_unit_metric",
            ],
        )
        
        # Process the CPU ID
        df["cpu_id"] = df["cpu_id"].str.removeprefix("CPU").astype(int)
        
        # Handle '<not supported>' values in the 'value' column
        mask = df["value"] == "<not supported>"
        if mask.any():
            logger.warning(f"Found {mask.sum()} '<not supported>' values in {stat_output_path}, will set them to NaN")
            df.loc[mask, "value"] = float('nan')
            
        # Handle '<not counted>' values in the 'value' column
        mask_not_counted = df["value"] == "<not counted>"
        if mask_not_counted.any():
            logger.warning(f"Found {mask_not_counted.sum()} '<not counted>' values in {stat_output_path}, will set them to NaN")
            df.loc[mask_not_counted, "value"] = float('nan')
        
        # Convert columns to appropriate types
        df = df.astype(
            {
                "timestamp": "float64",
                "value": "float64",  # Use float64 instead of int64 to handle NaN
                "unit": str,
                "metric_type": str,
                "run_time(ns)": "int64",
                "run_percentage": "float64",
                "opt_value": "float64",
                "opt_unit_metric": str,
            }
        )
        return df

    def get_available_events(self) -> List[str]:
        """Get all available events in the data.

        Returns:
            List[str]: list of avaiable events.
        """
        if self.use_ef_cores:
            # Return extracted event names
            all_metrics = self.data["metric_type"].unique()
            return list(set([self._extract_core_event(m) for m in all_metrics]))
        else:
            df = self.get_wider_data()
            col = df.columns.copy()
            col = col.drop(["timestamp", "cpu_id"])
            return col.to_list()

    def plot_interactive_event(
        self,
        events: Optional[List[str]] = None,
        threads: Optional[List[int]] = None,
        aggregation: bool = False,
        raw_data: bool = False,
        show: bool = True,
        write_to_html: Optional[str] = None,
    ) -> List[go.Scatter]:
        df = self.get_wider_data()
        scatters = []
        if threads:
            df = df[df["cpu_id"].isin(threads)]
        if aggregation:
            df = df.groupby(["timestamp"]).mean(numeric_only=True).reset_index()
            df["cpu_id"] = "all"
        # prevent duplicated events
        if events:
            events = list(set(events))
        else:
            events = df.columns.copy()
            events = events.drop(["timestamp", "cpu_id"])
        avail_threads = df["cpu_id"].unique().tolist()
        for t in avail_threads:
            data = df[df["cpu_id"] == t]
            for i, y in enumerate(events):
                r, g, b = generate_unique_rgb_color([t, i], generate_seed=True)
                try:
                    scatters.append(
                        go.Scatter(
                            x=data["timestamp"],
                            y=data[y],
                            mode="lines+markers",
                            name=f"CPU {t} {y}",
                            # different colors
                            line=dict(color=f"rgb({r}, {g}, {b})"),
                        )
                    )
                except KeyError as e:
                    logger.warning(f"Not found event: {y} in the stat data: {e}")
        if raw_data:
            return scatters
        else:
            return make_single_plot(
                scatters=scatters,
                title="Stat events trend",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                show=show,
                write_to_html=write_to_html,
            )
