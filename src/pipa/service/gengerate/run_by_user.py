import questionary
from rich import print
from pipa.service.gengerate.common import (
    quest_basic,
    CORES_ALL,
    write_title,
    opener,
    parse_perf_data,
    move_old_file,
    CPU_CORE_TYPES,
    HAS_HYBRID_CORES,
    get_core_type_range_string,
)
from pipa.service.export_sys_config import write_export_config_script
import os


def quest():
    """
    Asks the user for configuration options related to perf-record and perf-stat runs.

    Returns:
        dict: A dictionary containing the configuration options.
    """
    config = quest_basic()

    set_record = questionary.select(
        "Whether to set the duration of the perf-record run?\n",
        choices=["Yes", "No, I'll control it by myself. (Exit by Ctrl+C)"],
    ).ask()

    duration_record, duration_stat = None, None

    if set_record == "Yes":
        duration_record = questionary.text(
            "How long do you want to run perf-record? (Default: 120s)\n", "120"
        ).ask()

    stat_tool = "emon" if config["use_emon"] else "perf-stat"
    set_stat = questionary.select(
        f"Whether to set the duration of the {stat_tool} run?\n",
        choices=["Yes", "No, I'll control it by myself. (Exit by Ctrl+C)"],
    ).ask()
    if set_stat == "Yes":
        duration_stat = questionary.text(
            "How long do you want to run perf-stat? (Default: 120s)\n", "120"
        ).ask()

    # Add question for hybrid cores mode
    hybrid_cores = questionary.select(
        "Use hybrid cores mode (separate events for P-cores and E-cores)?\n", 
        choices=["Yes", "No"], 
        default="No"
    ).ask()
    
    config["hybrid_cores"] = hybrid_cores == "Yes"
    
    config["duration_record"] = duration_record
    config["duration_stat"] = duration_stat

    return config


def generate(config: dict):
    """
    Generate shell scripts for collecting and parsing performance data.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        None
    """
    workspace = config["workspace"]
    freq_record = config["freq_record"]
    events_record = config["events_record"]
    stat_time = config.get("duration_stat", 120)
    duration_record = config.get("duration_record", 120)
    if duration_record == -1:
        duration_record = None
    if stat_time == -1:
        stat_time = None
    annotete = config.get("annotete", False)
    use_emon = config.get("use_emon", False)
    # Use the hybrid_cores parameter from config instead of the global variable
    hybrid_cores = config.get("hybrid_cores", False)

    if use_emon:
        mpp = config.get("MPP_HOME", config.get("mpp", None))
    else:
        events_stat = config["events_stat"]
        count_delta_stat = config.get("count_delta_stat", 1000)

    if not os.path.exists(workspace):
        os.makedirs(workspace)

    with open(os.path.join(workspace, "pipa-collect.sh"), "w", opener=opener) as f:
        write_title(f)

        f.write("WORKSPACE=" + workspace + "\n")
        move_old_file(f)
        f.write("mkdir -p $WORKSPACE\n\n")

        f.write("ps -aux -ef --forest > $WORKSPACE/ps.txt\n")

        f.write(
            f"perf record -e '{events_record}' -g -a -F"
            + f" {freq_record} -o $WORKSPACE/perf.data"
            + (f" -- sleep {duration_record}\n" if duration_record else "\n")
        )

        f.write("sar -o $WORKSPACE/sar.dat 1 >/dev/null 2>&1 &\n")
        f.write("sar_pid=$!\n")
        if use_emon:
            f.write(
                f"emon -i {mpp}/emon_event_all.txt -v -f $WORKSPACE/emon_result.txt -t 0.1 -l 100000000 -c -experimental "
                + (f"-w sleep {stat_time}\n" if stat_time else "\n")
            )
        else:
            # For hybrid architecture with P-cores and E-cores
            if hybrid_cores:
                # Run perf on P-cores
                p_cores_range = get_core_type_range_string("p_cores")
                # Prefix each event with cpu_core/ for P-cores
                p_core_events = ",".join([f"cpu_core/{event}/" for event in events_stat.split(",")])
                f.write(
                    f"perf stat -e {p_core_events} -C {p_cores_range} -A -x , -I {count_delta_stat} -o $WORKSPACE/perf-stat-pcores.csv"
                    + (f" sleep {stat_time} &\n" if stat_time else " &\n")
                )
                f.write("p_cores_pid=$!\n")
                
                # Run perf on E-cores
                e_cores_range = get_core_type_range_string("e_cores")
                # Prefix each event with cpu_atom/ for E-cores
                e_core_events = ",".join([f"cpu_atom/{event}/" for event in events_stat.split(",")])
                f.write(
                    f"perf stat -e {e_core_events} -C {e_cores_range} -A -x , -I {count_delta_stat} -o $WORKSPACE/perf-stat-ecores.csv"
                    + (f" sleep {stat_time}\n" if stat_time else "\n")
                )
                
                # Wait for the P-cores perf to finish
                f.write("wait $p_cores_pid\n")
                
                # Combine the results - skip first two lines of the second file
                f.write("cat $WORKSPACE/perf-stat-pcores.csv <(tail -n +3 $WORKSPACE/perf-stat-ecores.csv) > $WORKSPACE/perf-stat.csv\n")
            else:
                # Use the original approach for homogeneous cores
                f.write(
                    f"perf stat -e {events_stat} -C {CORES_ALL[0]}-{CORES_ALL[-1]} -A -x , -I {count_delta_stat} -o $WORKSPACE/perf-stat.csv"
                    + (f" sleep {stat_time}\n" if stat_time else "\n")
                )
        f.write("kill -9 $sar_pid\n")

        f.write("echo 'Performance data collected successfully.'\n")

    with open(os.path.join(workspace, "pipa-parse.sh"), "w", opener=opener) as f:
        write_title(f)
        f.write("WORKSPACE=" + workspace + "\n")

        parse_perf_data(f)

        f.write("LC_ALL='C' sar -A -f $WORKSPACE/sar.dat >$WORKSPACE/sar.txt\n\n")

        if use_emon:
            f.write(
                f"python {mpp}/mpp/mpp.py -i $WORKSPACE/emon_result.txt -m {mpp}/metrics/icelake_server_2s_nda.xml -o ./ --thread-view"
            )

        if annotete:
            f.write(
                "perf annotate -i $WORKSPACE/perf.data > $WORKSPACE/perf.annotate\n\n"
            )

        write_export_config_script(f, os.path.join(workspace, "config"))

        f.write("echo 'Performance data parsed successfully.'\n")

        print("Shell script generated successfully.")
        print(
            f"Please check the script in {workspace}/pipa-collect.sh and {workspace}/pipa-parse.sh"
        )
        print(
            "Note you need to make sure the workload is running when you call pipa-collect.sh",
            "and the workload is finished when you call pipa-parsed.sh.",
            "Otherwise, the performance data may be incomplete or incorrect."
            "You should ensure that the total workload is longer than ten minutes."
            "The core list is generated according to the machine which runs this script now.",
            "Please check the configuration file for more details.",
        )


def main():
    generate(quest())


if __name__ == "__main__":
    main()
