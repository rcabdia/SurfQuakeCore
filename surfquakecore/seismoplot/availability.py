#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
availability
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from obspy import read
from datetime import datetime
import concurrent.futures
import platform
import matplotlib as mplt
import time
import os

class PlotExplore:

    @staticmethod
    def process_waveform_file(file_path):
        try:
            start_t = time.time()
            header = read(file_path, headonly=True)
            gap = header.get_gaps()
            net = header[0].stats.network
            sta = header[0].stats.station
            chn = header[0].stats.channel
            name = f"{net}.{sta}.{chn}"
            start = header[0].stats.starttime
            end = header[0].stats.endtime
            gap_intervals = [(g[4], g[5]) for g in gap]
            duration = end - start
            elapsed = time.time() - start_t

            return {
                'name': name,
                'start': start,
                'end': end,
                'gaps': gap_intervals,
                'file': os.path.basename(file_path),
                'duration': duration,
                'elapsed': elapsed
            }
        except Exception as e:
            return {'error': str(e), 'file': os.path.basename(file_path)}

    @staticmethod
    def data_availability_new(list_files: list):
        if platform.system() == 'Darwin':
            mplt.use("MacOSX")
        else:
            mplt.use("Qt5Agg")

        fig, hax = plt.subplots(1, 1, figsize=(12, 6))

        starttimes = []
        endtimes = []

        print(f"🔍 Starting processing of {len(list_files)} waveform files...\n")
        start_overall = time.time()

        # Process in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(PlotExplore.process_waveform_file, f): f for f in list_files}
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                result = future.result()
                results.append(result)

                if 'error' in result:
                    print(f"Failed [{idx}/{len(list_files)}] Error processing {result['file']}: {result['error']}")
                else:
                    print(f"OK [{idx}/{len(list_files)}] Processed {result['file']} "
                          f"({result['name']}) - Duration: {result['duration']:.1f}s - "
                          f"Read time: {result['elapsed']:.2f}s")

        print(f"\nOK Finished all files in {time.time() - start_overall:.2f} seconds.\n")

        year_ranges = defaultdict(list)

        for res in results:
            if 'error' in res:
                continue

            name = res['name']
            start = res['start'].matplotlib_date
            end = res['end'].matplotlib_date
            hax.hlines(name, start, end, colors='k', linestyles='solid', lw=2)

            starttimes.append(res['start'])
            endtimes.append(res['end'])

            year_ranges[res['start'].year].append((res['start'], res['end']))

            for gap_start, gap_end in res['gaps']:
                hax.hlines(name, gap_start.matplotlib_date, gap_end.matplotlib_date,
                           colors='r', linestyles='solid', lw=2)

        if starttimes and endtimes:
            start_time = min(starttimes)
            end_time = max(endtimes)

            formatter = mdt.DateFormatter('%m/%d %H:%M')
            hax.xaxis.set_major_formatter(formatter)
            hax.set_xlabel("Date")
            hax.set_xlim(start_time.matplotlib_date, end_time.matplotlib_date)

            plt.setp(hax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

            # Vertical lines and year labels
            years = range(start_time.year, end_time.year + 1)
            for y in years:
                dt_line = datetime(y, 1, 1)
                x_line = mdt.date2num(dt_line)
                hax.axvline(x_line, color='gray', linestyle='--', lw=1)

                # Text label only if there’s data that year
                if y in year_ranges:
                    starts = [r[0].matplotlib_date for r in year_ranges[y]]
                    ends = [r[1].matplotlib_date for r in year_ranges[y]]
                    mid_x = (min(starts) + max(ends)) / 2

                    hax.text(
                        mid_x, 1.01,
                        str(y),
                        transform=hax.get_xaxis_transform(),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color='gray',
                        fontweight='bold'
                    )

        plt.tight_layout()
        plt.show(block=True)