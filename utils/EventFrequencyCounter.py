import pandas as pd


class EventFrequencyCounter:
    def __init__(self, period='1S', print_periodic_reports=False):
        self.event_times = []
        self.periodic_report = True
        self.period = period
        self.this_period_events = []
        self.print_periodic_reports = print_periodic_reports

    def event_occurrence(self):
        cur_time = pd.Timestamp.now()
        self.event_times.append(cur_time)
        if self.print_periodic_reports:
            if self.this_period_events and self.this_period_events[-1].floor(self.period) != cur_time.floor(self.period):
                r = self.report(periodic=True)
                print(f'{r.index[0]} FPS:  {r.iloc[0]}')
                self.this_period_events = []
            self.this_period_events.append(cur_time)

    def report(self, periodic=False):
        df = pd.DataFrame({'time': self.event_times if not periodic else self.this_period_events})
        return df.groupby(pd.Grouper(key='time', freq=self.period))['time'].count()
