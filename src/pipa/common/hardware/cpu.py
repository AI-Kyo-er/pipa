from pipa.common.cmd import run_command
from psutil import cpu_count
import os


def get_cpu_cores():
    """
    Returns a list of the number of CPU cores.

    This function uses the `lscpu` command to retrieve the number of CPU cores
    on the system. It parses the output of the command and returns a list of
    integers representing the number of CPU cores.

    Returns:
        list: A list of integers representing the number of CPU cores.

    Example:
        >>> get_cpu_cores()
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    cpu_list = [
        l
        for l in run_command("lscpu -p=cpu", log=False).split("\n")
        if not l.startswith("#")
    ]
    return [int(x) for x in cpu_list]


def get_cpu_core_types():
    """
    Detects performance (P) cores and efficiency (E) cores in hybrid CPU architectures.
    
    This function checks for the existence of /sys/devices/cpu_core/cpus and
    /sys/devices/cpu_atom/cpus files which indicate a hybrid architecture with
    different core types. If these files are found, it reads them to determine
    which CPU threads belong to P-cores and which belong to E-cores.
    
    Returns:
        dict: A dictionary with keys 'p_cores' and 'e_cores', each containing a list
              of CPU thread IDs. If the system does not have distinct core types,
              all cores are listed under 'p_cores' and 'e_cores' is empty.
              
    Example:
        >>> get_cpu_core_types()
        {'p_cores': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
         'e_cores': [12, 13, 14, 15, 16, 17, 18, 19]}
    """
    result = {'p_cores': [], 'e_cores': []}
    
    # Check if the system has hybrid architecture with P and E cores
    if os.path.exists("/sys/devices/cpu_core") and os.path.exists("/sys/devices/cpu_atom"):
        try:
            # Read P-cores
            with open("/sys/devices/cpu_core/cpus", "r") as f:
                p_cores_str = f.read().strip()
                result['p_cores'] = parse_cpu_range(p_cores_str)
                
            # Read E-cores
            with open("/sys/devices/cpu_atom/cpus", "r") as f:
                e_cores_str = f.read().strip()
                result['e_cores'] = parse_cpu_range(e_cores_str)
                
            return result
        except Exception as e:
            # If any error occurs, fall back to treating all cores as P-cores
            pass
    
    # If not a hybrid architecture or error occurred, treat all cores as P-cores
    result['p_cores'] = get_cpu_cores()
    return result


def parse_cpu_range(range_str):
    """
    Parse CPU range string like "0-7" or "0-3,8-11" into a list of integers.
    
    Args:
        range_str (str): String representation of CPU ranges
        
    Returns:
        list: List of CPU IDs
    """
    result = []
    if not range_str:
        return result
        
    ranges = range_str.split(',')
    for r in ranges:
        if '-' in r:
            start, end = map(int, r.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(r))
    
    return result


NUM_CORES_PHYSICAL = cpu_count(logical=False)  # Number of physical cores
