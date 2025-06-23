#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_command_prompt.py
"""
import os
import readline
import atexit

class PlotCommandPrompt:
    def __init__(self, plot_proj):
        self.plot_proj = plot_proj
        self.prompt_active = False
        histfile = os.path.expanduser("~/.plot_command_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, histfile)
        self.commands = {
            "n": self._cmd_next,
            "b": self._cmd_prev,
            "spectrogram": self._cmd_spectrogram,
            "spec": self._cmd_spectrogram,
            "spectrum": self._cmd_spectrum,
            "sp": self._cmd_spectrum,
            "cwt": self._cmd_cwt,
            "p": self._cmd_pick,
            "fk": self._cmd_fk,
            "plot_type": self._cmd_type,
            "cut": self._cmd_cut,
            "concat": self._cmd_concat,
            "shift": self._cmd_shift,
            "write": self._cmd_write,
            "help": self._cmd_help,
            "exit": self._cmd_exit
        }

    def _cmd_exit(self, args):
        """
        Exit the plot and close the interactive session.
        Usage: exit
        """
        import matplotlib.pyplot as plt

        print("[INFO] Exiting interactive plotting session.")
        self.prompt_active = False
        self._exit_code = "exit"

        try:
            plt.close(self.plot_proj.fig)
        except Exception:
            pass  # In case fig is None or already closed

    def run(self) -> str:
        """
        Run the interactive command prompt loop.

        Returns
        -------
        str
            A string exit code:
            - 'next': User typed 'n' to continue
            - 'pick': User wants to return to picking mode
        """
        
        if self.prompt_active:
            return "next"  # Prevent re-entry

        self.prompt_active = True
        self._exit_code = "next"  # Default: move to next plot

        while self.prompt_active:
            try:
                cmd_line = input(">> ").strip()
                if not cmd_line:
                    continue

                parts = cmd_line.split()
                cmd = parts[0].lower()

                handler = self.commands.get(cmd)
                if handler:
                    handler(parts)
                else:
                    print(f"[WARN] Unknown command: {cmd}")
            except Exception as e:
                print(f"[ERROR] Command failed: {e}")

        return self._exit_code

    def _cmd_pick(self, args):
        print("[INFO] Returning to picking mode...")
        self.prompt_active = False
        self._exit_code = "p"

    def _cmd_next(self, args):
        print("[INFO] Advancing to next subplot set...")
        self.plot_proj.current_page += 1  # advance page
        self.plot_proj.clear_plot()  # clear figure
        self.plot_proj.plot(page=self.plot_proj.current_page)  # replot
        # Do not set prompt_active = False; stay in prompt!

    def _cmd_spectrogram(self, args):
        if len(args) >= 2:
            idx = int(args[1])
            win = float(args[2]) if len(args) > 2 else 5.0
            overlap = float(args[3]) if len(args) > 3 else 50.0
            self.plot_proj._plot_spectrogram(idx, win, overlap)
        else:
            print("Usage: spectrogram <index> [<win_sec> <overlap%>]")

    def _cmd_spectrum(self, args):
        if len(args) < 2 or len(args) > 3:
            print("Usage: spectrum <index>|all [axis_type]")
            return

        target = args[1]
        axis_type = args[2] if len(args) == 3 else 'loglog'

        if target == "all":
            self.plot_proj._plot_all_spectra(axis_type=axis_type)
        elif target.isdigit():
            self.plot_proj._plot_single_spectrum(int(target), axis_type=axis_type)
        else:
            print("[ERROR] Invalid spectrum command")

    def _cmd_cwt(self, args):
        if len(args) not in [4, 6]:
            print("Usage: cwt <index> <wavelet> <param> [<fmin> <fmax>]")
            return
        idx = int(args[1])
        wavelet = args[2]
        param = float(args[3])
        if len(args) == 6:
            fmin = float(args[4])
            fmax = float(args[5])
            self.plot_proj._plot_wavelet(idx, wavelet, param, fmin=fmin, fmax=fmax)
        else:
            self.plot_proj._plot_wavelet(idx, wavelet, param)

    def _cmd_fk(self, args):
        """
        Run FK analysis and show output.
        Usage: fk [--fmin 0.8] [--fmax 2.2] [--smax 0.3] [--grid 0.05] [--win 3] [--overlap 0.1]
        """
        import matplotlib.pyplot as plt

        # Default parameters
        params = {
            "fmin": 0.8,
            "fmax": 2.2,
            "smax": 0.3,
            "slow_grid": 0.05,
            "timewindow": 3,
            "overlap": 0.05
        }

        # Parse command-line style args
        it = iter(args[1:])  # Skip 'fk'
        for arg in it:
            if arg.startswith("--"):
                key = arg[2:]
                try:
                    val = float(next(it))
                    if key == "grid":
                        params["slow_grid"] = val
                    else:
                        params[key] = val
                except (StopIteration, ValueError):
                    print(f"[ERROR] Invalid value for --{key}")
                    return

        print(f"[INFO] Running FK with parameters: {params}")
        try:
            self.plot_proj._run_fk(**params)
        except Exception as e:
             print(f"[ERROR] FK run failed: {e}")

    def _cmd_prev(self, args):
        if self.plot_proj.current_page > 0:
            print("[INFO] Returning to previous subplot set...")
            self.plot_proj.current_page -= 1
            self.plot_proj.clear_plot()
            self.plot_proj.plot(page=self.plot_proj.current_page)
        else:
            print("[INFO] Already at the first page.")

    def _cmd_type(self, args):
        """
        Change the plotting mode.
        Usage: mode <plot_mode>
        Example: mode time | mode filtered | mode summary
        """
        if len(args) != 2:
            print("Usage: mode <plot_type>")
            return

        new_mode = args[1].lower()
        if new_mode not in self.plot_proj.available_modes:
            print(f"[ERROR] Unknown mode '{new_mode}'. Available plot types: {', '.join(self.plot_proj.available_types)}")
            return

        self.plot_proj.plot_config["plot_type"] = new_mode
        print(f"[INFO] Plot type changed to '{new_mode}'")
        self.plot_proj.clear_plot()
        self.plot_proj.plot(page=self.plot_proj.current_page)

    def _cmd_write(self, args):

        """
        Write currently displayed traces to disk in HDF5 format.

        Usage:
            write --folder_path <output_folder>
        """

        import os
        folder_path = None

        # Parse argument for folder path
        it = iter(args[1:])
        for arg in it:
            if arg == "--folder_path":
                try:
                    folder_path = next(it)
                except StopIteration:
                    print("[ERROR] Missing value after --folder_path")
                    return

        if not folder_path:
            print("[ERROR] --folder_path must be specified")
            return

        # Ensure the output directory exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"[INFO] Created folder: {folder_path}")
            except Exception as e:
                print(f"[ERROR] Failed to create folder '{folder_path}': {e}")
                return

        # Get displayed traces only
        displayed_traces = getattr(self.plot_proj, "displayed_traces", None)
        if not displayed_traces:
            print("[WARN] No traces currently displayed to write.")
            return

        try:
            self._write_files(displayed_traces, folder_path)
        except Exception as e:
            print(f"[ERROR] Failed to write displayed traces: {e}")

    def _write_files(self, stream, output_folder):
        """
        Write each trace in the stream to a unique file in HDF5 format.

        Parameters
        ----------
        stream : obspy.Stream
            Stream to be written.
        output_folder : str
            Folder where files should be written.
        """
        import os

        errors = False

        for tr in stream:
            try:
                t1 = tr.stats.starttime
                base_name = f"{tr.id}.D.{t1.year}.{t1.julday}"
                path_output = os.path.join(output_folder, base_name)

                counter = 1
                while os.path.exists(path_output + ".h5"):
                    path_output = os.path.join(output_folder, f"{base_name}_{counter}")
                    counter += 1

                path_output += ".h5"

                print(f"[INFO] {tr.id} - Writing to {path_output}")
                tr.write(path_output, format="H5")

            except Exception as e:
                errors = True
                print(f"[ERROR] Could not write {tr.id}: {e}")

        if errors:
            print(f"[WARN] Writing finished with some errors.")
        else:
            print(f"[INFO] All traces written successfully to: {output_folder}")

    from obspy import UTCDateTime
    from datetime import datetime

    def _cmd_cut(self, args):

        from datetime import datetime
        from obspy import UTCDateTime

        """
        Cut traces based on a phase, reference, or absolute UTC start/end.

        Usage:
            cut --phase <name> <t_before> <t_after>
            cut --reference <t_before> <t_after>
            cut --start "YYYY-MM-DD HH:MM:SS" --end "YYYY-MM-DD HH:MM:SS"
        """

        new_traces = []

        # Case 1: Absolute time cut
        if "--start" in args and "--end" in args:
            try:
                start_str = args[args.index("--start") + 1]
                end_str = args[args.index("--end") + 1]
                t1 = UTCDateTime(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S"))
                t2 = UTCDateTime(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S"))
            except Exception as e:
                print(f"[ERROR] Invalid date format: {e}")
                return

            for tr in self.plot_proj.trace_list:
                try:
                    tr_cut = tr.copy().trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
                    new_traces.append(tr_cut)
                except Exception as e:
                    print(f"[ERROR] Could not cut {tr.id}: {e}")

        # Case 2: Phase-based cut
        elif "--phase" in args:
            try:
                i = args.index("--phase")
                phase_name = args[i + 1]
                t_before = float(args[i + 2])
                t_after = float(args[i + 3])
            except (IndexError, ValueError):
                print("[ERROR] Usage: cut --phase <name> <t_before> <t_after>")
                return

            for tr in self.plot_proj.trace_list:
                phase_time = None

                if hasattr(tr.stats, "picks"):
                    matching_picks = [
                        pick for pick in tr.stats.picks
                        if pick.get("phase", "").lower() == phase_name.lower()
                    ]
                    if matching_picks:
                        # Take the last one
                        phase_time = UTCDateTime(float(matching_picks[-1]["time"]))

                if phase_time is None:
                    print(f"[WARN] Trace {tr.id} missing phase '{phase_name}' — skipped.")
                    continue

                try:
                    phase_time = UTCDateTime(phase_time)
                    t1 = phase_time - t_before
                    t2 = phase_time + t_after
                    tr_cut = tr.copy().trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
                    new_traces.append(tr_cut)
                except Exception as e:
                    print(f"[ERROR] Could not cut {tr.id}: {e}")

        # Case 3: Reference-based cut
        elif "--reference" in args:
            try:
                i = args.index("--reference")
                t_before = float(args[i + 1])
                t_after = float(args[i + 2])
            except (IndexError, ValueError):
                print("[ERROR] Usage: cut --reference <t_before> <t_after>")
                return

            ref_time = getattr(self.plot_proj, "last_reference", None)
            if not ref_time:
                print("[ERROR] No reference time found. Cannot cut.")
                return

            try:

                t1 = ref_time - t_before
                t2 = ref_time + t_after
            except Exception as e:
                print(f"[ERROR] Invalid reference time: {e}")
                return

            for tr in self.plot_proj.trace_list:
                try:
                    tr_cut = tr.copy().trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
                    new_traces.append(tr_cut)
                except Exception as e:
                    print(f"[ERROR] Could not cut {tr.id}: {e}")

        else:
            print("[ERROR] You must specify --phase, --reference, or --start/--end")
            return

        if not new_traces:
            print("[WARN] No traces were successfully cut.")
            return

        # Apply cuts to plot_proj
        self.plot_proj.trace_list = new_traces
        self.plot_proj.current_page = 0
        self.plot_proj.clear_plot()
        self.plot_proj.plot(page=0)
        print(f"[INFO] Cutting complete. {len(new_traces)} traces updated and replotted.")

    def _cmd_concat(self, args):
        """
        Concatenate traces by station/component using ObsPy's merge method.

        Usage:
            concat
        Notes:
            - This applies to all traces in trace_list.
            - Overlapping segments are handled with default 'interpolate'.
        """
        from obspy import Stream

        try:
            st = Stream(self.plot_proj.trace_list)
            print(f"[INFO] Merging {len(st)} trace segments...")

            st.merge(method=1, fill_value='interpolate')  # method=1 = interpolate across gaps
            self.plot_proj.trace_list = list(st)

            # Reset plot
            self.plot_proj.current_page = 0
            self.plot_proj.clear_plot()
            self.plot_proj.plot(page=0)

            print(f"[INFO] Concatenation complete. Now {len(st)} traces after merge.")

        except Exception as e:
            print(f"[ERROR] Failed to concatenate traces: {e}")

    def _cmd_shift(self, args):
        """
        Align traces by phase pick time or theoretical arrival.

        Usage:
            shift --phase <phase_name>
            shift --phase_theo <phase_name>
        """
        from obspy import UTCDateTime

        # --- Determine phase argument type ---
        phase_name = None
        phase_source = None
        for key in ["--phase", "--phase_theo"]:
            if key in args:
                try:
                    idx = args.index(key)
                    phase_name = args[idx + 1]
                    phase_source = key
                    break
                except (IndexError, ValueError):
                    print(f"[ERROR] Usage: shift {key} <phase_name>")
                    return

        if not phase_name:
            print("[ERROR] You must specify --phase or --phase_theo <phase_name>")
            return

        shifted_count = 0

        # --- Process each trace ---
        for tr in self.plot_proj.trace_list:
            pick_time = None

            if phase_source == "--phase":
                picks = getattr(tr.stats, "picks", [])
                for pick in picks:
                    if pick.get("phase") == phase_name:
                        pick_time = pick.get("time")  # already a timestamp
                        break

            elif phase_source == "--phase_theo":
                arrivals = tr.stats.get("geodetic", {}).get("arrivals", [])
                for arr in arrivals:
                    if arr.get("phase") == phase_name:
                        pick_time = arr.get("time")  # timestamp expected
                        break

            if pick_time is not None:
                try:
                    pick_time_utc = UTCDateTime(pick_time)
                    shift_amount = pick_time_utc - tr.stats.starttime
                    tr.stats.starttime = UTCDateTime(0)  # Align all to zero
                    samples_to_shift = int(shift_amount / tr.stats.delta)

                    if samples_to_shift >= len(tr.data):
                        print(f"[WARN] Shift exceeds trace length for {tr.id} — skipping.")
                        continue

                    tr.data = tr.data[samples_to_shift:]  # Rough shift
                    shifted_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to shift {tr.id}: {e}")
            else:
                print(f"[WARN] Phase '{phase_name}' not found in {tr.id}")

        if shifted_count == 0:
            print("[WARN] No traces were shifted.")
            return

        # Replot from beginning
        self.plot_proj.current_page = 0
        self.plot_proj.clear_plot()
        self.plot_proj.plot(page=0)
        print(f"[INFO] Shifted {shifted_count} traces by phase '{phase_name}' and replotted.")

    def _cmd_help(self, args):

        print("Available commands:")
        print("  p                             Return to interactive picking mode")
        print("  n                             Next set of traces / exit prompt")
        print("  b                             Previous set of traces")
        print("  spectrum <index>|all [type]  Plot amplitude spectrum (type: loglog, xlog, ylog).")
        print("                               Example: >> sp 0 | >> sp all ylog")
        print("  spec <idx> [win overlap]     Plot spectrogram for one trace.")
        print("                               Example: >> spec 0 5.0 50")
        print("  cwt <idx> <wavelet> <param> [fmin fmax]  Continuous wavelet transform.")
        print("                               Wavelets: cm (Morlet), mh, pa")
        print("                               Example: >> cwt 0 cm 6 0.5 8")
        print("  fk [--fmin <Hz>] [--fmax <Hz>] [--smax <s/km>] [--grid <step>] [--win <s>] [--overlap <ratio>]")
        print("                               Run FK analysis. Example: >> fk --fmin 0.8 --fmax 2.0")
        print("  plot_type <type>             Change plot mode. Types: standard, record, overlay")
        print("                               Example: >> plot_type overlay")
        print("  concat                           Merge/concatenate traces with same ID using ObsPy")
        print("  shift --phase <name>              Shift traces to align by phase pick (e.g., 'P')")
        print("  shift --phase_theo <name>          Shift traces to align by theretical phase (e.g., 'P')")
        print("  cut --phase <name> <before> <after>     Trim traces using phase picks (e.g., 'P', 'S')")
        print("  cut --reference <before> <after>        Trim traces using last reference time")
        print("  write --folder_path <path>   Write current traces to folder in HDF5 format.")
        print("                               Example: >> write --folder_path ./output")
        print("  exit                            Close plot and exit interactive prompt")
        print("  help                          Show this help message")