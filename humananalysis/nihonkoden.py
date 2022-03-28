import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import interpolate
from scipy import integrate


class Nihonkoden():
    '''日本光電の出力データに対応する分析プログラム
    日本光電に出力データの読み込みから,解析,データ表示を行う
    '''

    def __init__(self, DataPath: str = "") -> None:

        with open(DataPath, 'r', encoding='cp932', newline='', errors="replace") as f:
            lines = list(f.read().splitlines())

            for i, line in enumerate(lines[:20]):
                if line.startswith("Interval"):
                    if line.split('=')[-1] == "1.0ms":
                        self.Interval = 0.001  # 1ms
                    else:
                        print("Enter Interval time[ms]")
                        self.Interval = float(input())/1000

                if line.startswith("#Address"):
                    self.Address = list(line.split('=')[-1].split(','))  # ex. [A1,A2,A3,A5,Event]
                    self.Address[-1] = "Event"
                elif line.startswith("ChannelTitle"):
                    self.ChannelTitle = list(line.split('=')[-1].split(','))
                    self.ChannelTitle[-1] = "Event"
                    # print(self.ChannelTitle)

                elif line.startswith("#Data="):
                    DataStartIdx = i
                    pass

        df = pd.read_csv(DataPath, header=None, sep=',', names=self.Address, encoding='cp932', skiprows=DataStartIdx+1, dtype="object")
        self.DataFrame = df

        self.EventsDF = df[df['Event'].str.startswith('#*', na=False)]['Event']

        self.MarkersDF = df[df['Event'].str.startswith('#* M', na=False)]['Event']

        return

    def calLFHF(
            self, starts: list = [],
            ends: list = [],
            prominence: float = 1, height: float = None, re_sampling_freq: float = 1,
            plot: bool = True, plot_vorbose: bool = False, revECG: bool = False) -> list:

        if len(starts) != len(ends):
            raise Exception('starts and ends are must be same length')
        LFHFlist = []
        for start, end in zip(starts, ends):
            ECGSignal = self.DataFrame["A5"].iloc[start:end].astype('float').values
            if revECG:
                ECGSignal = -ECGSignal
            # 以下岡野LFHF3を使用
            peak_index_array, _ = find_peaks(ECGSignal, prominence=prominence, height=height)
            time = np.arange(0, len(ECGSignal)*self.Interval, self.Interval)
            peak_time_array = peak_index_array*self.Interval

            if plot:
                plt.figure()
                plt.plot(time, ECGSignal, label="ECGSignal")
                plt.plot(peak_time_array, ECGSignal[peak_index_array], "ob", label="peaks")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title("peaks on ECGsignal")
                plt.tight_layout()
                # plt.xlim([1000,3000])
                plt.show()

            rr_intervals = (peak_time_array[1:]-peak_time_array[:-1])
            rr_times = peak_time_array[:-1]
            if plot_vorbose:
                plt.figure(figsize=(10, 5))
                plt.plot(rr_times, rr_intervals)
                plt.grid(True)
                # plt.ylim(0, )
                plt.show()

            # print(f"{rr_times=}")

            N = int(np.floor(rr_times[-1]-rr_times[0]/re_sampling_freq))  # re_sampling_freq で何点取れるか

            # print(N)

            f = interpolate.interp1d(rr_times, rr_intervals, kind='linear')

            t = np.arange(0, N, 1)
            y = f(rr_times[0]+t)
            if plot_vorbose:
                plt.figure()
                plt.plot(t, y)
                plt.grid(True)
                # plt.ylim(0, )
                plt.show()

            Y = np.fft.fft(y)
            fq = np.fft.fftfreq(len(y), d=1/re_sampling_freq)
            Y_abs_amp = abs(Y*Y)

            if plot_vorbose:
                plt.plot(fq[:N//2], Y_abs_amp[:N//2])
                plt.ylim(0, 8)
                plt.show()

            start_index = np.where(fq > 0.05)[0][0]
            end_index = np.where(fq > 0.15)[0][0]
            LF = integrate.cumtrapz(Y_abs_amp[start_index:end_index], fq[start_index:end_index])

            start_index = np.where(fq > 0.15)[0][0]
            end_index = np.where(fq > 0.40)[0][0]
            HF = integrate.cumtrapz(Y_abs_amp[start_index:end_index], fq[start_index:end_index])

            print("LF/HFは", LF[-1]/HF[-1])
            LFHFlist.append(LF[-1]/HF[-1])

        return LFHFlist

    def showGraph(self, columns: list = [], setplot: bool = True, divplot: bool = True):

        df = self.DataFrame
        if columns == []:
            columns = self.Address[:-1]
        if setplot:
            _, axes = plt.subplots(len(columns), 1)
            for column, ax in zip(columns, axes):
                ax.plot(df.index * self.Interval, df[column].astype('float'), zorder=1)  # 筋電データ
                for idx in df[df['Event'].str.startswith('#* M', na=False)].index:
                    ax.axvline(idx * self.Interval, ls="-", color="red")
                ax.set_xlabel("time [sec]")
                ax.set_ylabel("amplified EMG [mV]")

        if divplot:
            for column in columns:
                plt.figure()
                plt.plot(df.index * self.Interval, df[column].astype('float'), zorder=1)  # 筋電データ
                for idx in df[df['Event'].str.startswith('#* M', na=False)].index:
                    plt.axvline(idx * self.Interval, ls="-", color="red")
                plt.xlabel("time [sec]")
                plt.ylabel("amplified EMG [mV]")

        plt.show()

        pass

    def getDataFrame(self, columns: list = []):
        df = self.DataFrame
        if columns == []:
            columns = self.Address[:-1]
        return df.loc[:, columns]

    def getRowSignal(self, channel=[], starts: list = [], ends: list = []) -> list:
        df = self.DataFrame
        signals = []

        for start, end in zip(starts, ends):
            signal = df[channel].iloc[start:end].astype(float).values
            signals.append(signal)
        return signals

    def calRootMeanSquareSignal(self, channels=[], starts: list = [], ends: list = [], window_time_ms: int = 50, plot: bool = False) -> list:
        def _rms(d): return np.sqrt((d ** 2).sum() / d.size)
        df = self.DataFrame
        signals = []
        for channel in channels:
            for start, end in zip(starts, ends):
                window_size = int(window_time_ms/(self.Interval*1000))
                # df["rms_signal"] = df[channel].iloc[start:end].rolling(window=window_size, min_periods=1, center=False).apply(_rms)
                # rms_signal = df["rms_signal"].values
                rms_signal = df[channel].iloc[start:end].rolling(window=window_size, min_periods=1, center=False).apply(_rms).values
                signals.append(rms_signal)
                if plot:
                    # print(len(rms_signal), start, end)
                    self.__dfplot(rms_signal, start, end)
        return signals

    def calMVC(self, channels=[], starts: list = [], ends: list = [], window_time_ms: int = 50, plot: bool = False) -> list:
        def _rms(d): return np.sqrt((d ** 2).sum() / d.size)
        df = self.DataFrame
        signals = []
        for channel in channels:
            for start, end in zip(starts, ends):
                window_size = int(window_time_ms/(self.Interval*1000))
                df["rms_signal"] = df[channel].iloc[start:end].rolling(window=window_size, min_periods=1, center=False).apply(_rms)
                rms_signal = df["rms_signal"].dropna().values
                # signals.append(rms_signal)
                signal = df["rms_signal"].iloc[start:end].rolling(window=3000, min_periods=1, center=False).apply(sum).values
                peak_index_array, _ = find_peaks(signal, prominence=30, height=None, distance=None)
                peak_index_array_r, _ = find_peaks(-signal, prominence=30, height=None, distance=None)
                plt.plot(range(len(signal)), signal)
                plt.plot(peak_index_array, signal[peak_index_array], "ob")
                plt.plot(peak_index_array_r, signal[peak_index_array_r], "ob")
                plt.show()

                # rms_time = np.arange(0, len(rms_signal)*self.Interval, self.Interval)
                # time = np.arange(0, len(signal)*self.Interval, self.Interval)
                # peak_time_array = peak_index_array*self.Interval
                # peak_time_array_r = peak_index_array_r*self.Interval
                starts = peak_index_array[0]
                print(f"{peak_index_array=}")
                ends = peak_index_array[0]+3000
                print(starts, ends)
                signals.append(rms_signal[starts-3000:ends-3000])
                if plot:
                    plt.figure()
                    # print(list(range(len(rms_signal)))[0:50])
                    print(rms_signal)
                    plt.plot(list(range(len(rms_signal))), rms_signal, label="-signal")
                    plt.plot(peak_index_array-3000, rms_signal[peak_index_array-3000], "ob", label="peaks")
                    plt.plot(peak_index_array_r-3000, rms_signal[peak_index_array_r-3000], "ob", label="peaks")
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.title("peaks on signal")
                    plt.tight_layout()
                    plt.show()
                    plt.clf()
                    plt.close()

        return signals

    def calEMGActiveStartEndTime(self, signals=[], prominence: float = 1, height: float = None, distance: float = 20, plot: bool = False) -> list:
        starts = []
        ends = []
        for signal in signals:
            signal = np.array(signal)
            peak_index_array, _ = find_peaks(-signal, prominence=prominence, height=height, distance=distance)
            time = np.arange(0, len(signal)*self.Interval, self.Interval)
            peak_time_array = peak_index_array*self.Interval
            starts.append([start for start in peak_index_array[:-1]])
            ends.append([end for end in peak_index_array[1:]])

            if plot:
                plt.plot(time, signal, label="-signal")
                plt.plot(peak_time_array, signal[peak_index_array], "ob", label="peaks")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title("peaks on signal")
                plt.tight_layout()
                plt.show()
        return starts, ends

    def calAverageAmplitude(self, signals=[], startslist=[[0]], endslist=[[-1]], plot: bool = False):
        average_amplitudeslist = []
        for signal, starts, ends in zip(signals, startslist, endslist):
            average_amplitudes = []
            for start, end in zip(starts, ends):
                # print(start, end)
                _signal = signal[start:end]
                # print(signal)
                # print(signal)
                if plot:
                    plt.plot(_signal)
                    plt.show()
                average_amplitude = sum(_signal)/len(_signal)
                average_amplitudes.append(average_amplitude)
            average_amplitudeslist.append(average_amplitudes)
        return average_amplitudeslist

    def calNumOfPeaks(
            self, signals=[],
            MVC: float = None, prominence: float = 1, height: float = None, distance: float = 20, plot: bool = False, threshold: float = None) -> list:
        num_of_peaks = []
        for signal in signals:
            peak_index_array, _ = find_peaks(signal, prominence=prominence, height=height, distance=distance)
            if plot:
                plt.plot(signal, label="-signal")
                plt.plot(peak_index_array, signal[peak_index_array], "ob", label="peaks")
                if MVC is not None:
                    plt.axhline(MVC)
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.title("peaks on signal")
                plt.tight_layout()
                plt.show()
            if threshold is not None:
                peak_index_array = [peak for peak in peak_index_array if peak >= threshold]
            num_of_peaks.append(len(peak_index_array))
        return num_of_peaks

    def __dfplot(self, signal, start, end):
        df = self.DataFrame
        plt.figure()
        plt.plot(df.iloc[start:end].index * self.Interval, signal, zorder=1)
        for idx in df.iloc[start:end][df.iloc[start:end]['Event'].str.startswith('#* M', na=False)].index:
            plt.axvline(idx * self.Interval, ls="-", color="red")
        plt.xlabel("time [msec]")
        plt.ylabel("amplified EMG [mV]")
        plt.show()


def main():

    pass


if __name__ == "__main__":

    main()