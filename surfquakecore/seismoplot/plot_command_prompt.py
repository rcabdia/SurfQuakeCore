#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_command_prompt.py
"""


class PlotCommandPrompt:
    def __init__(self, plot_proj):
        self.plot_proj = plot_proj
        self.prompt_active = False
        self.commands = {
            "q": self._cmd_next,
            "spectrogram": self._cmd_spectrogram,
            "spec": self._cmd_spectrogram,
            "spectrum": self._cmd_spectrum,
            "sp": self._cmd_spectrum,
            "cwt": self._cmd_cwt,
            "pick": self._cmd_pick,
            "fk": self._cmd_fk,
            "help": self._cmd_help
        }

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
        self._exit_code = "pick"

    def _cmd_next(self, args):
        self.prompt_active = False

    def _cmd_spectrogram(self, args):
        if len(args) >= 2:
            idx = int(args[1])
            win = float(args[2]) if len(args) > 2 else 5.0
            overlap = float(args[3]) if len(args) > 3 else 50.0
            self.plot_proj._plot_spectrogram(idx, win, overlap)
        else:
            print("Usage: spectrogram <index> [<win_sec> <overlap%>]")

    def _cmd_spectrum(self, args):
        if len(args) != 2:
            print("Usage: spectrum <index>|all")
            return
        arg = args[1]
        if arg == "all":
            self.plot_proj._plot_all_spectra()
        elif arg.isdigit():
            self.plot_proj._plot_single_spectrum(int(arg))
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
        #try:
        self.plot_proj._run_fk(**params)
        # except Exception as e:
        #     print(f"[ERROR] FK run failed: {e}")



    def _cmd_help(self, args):
        print("Available commands:")
        print("  pick                Return to interactive picking mode")
        print("  spectrum <idx|all>  Show amplitude spectrum")
        print("  spectrogram <idx> [win overlap]  Plot spectrogram")
        print("  cwt <idx> <wavelet> <param> [fmin fmax]  Plot wavelet")
        print("  n                   Next / exit prompt")