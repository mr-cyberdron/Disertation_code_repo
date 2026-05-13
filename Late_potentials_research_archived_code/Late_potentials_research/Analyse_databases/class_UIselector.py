import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pylab
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector


class UIselector:
    #
    def __init__(self):

        self.LVP = None
        self.LAP = None
        self.UNDO = None
        self.SAVE = None

        self.axs = None
        self.patch = None
        self.selected_ax = None
        self.span = None
        self.ax_codes = None
        self.ax_type = None
        self.fig = None
        self.selected_range = None
        self.position_sig_dict = None
        self.spans_list = []

        self.signals = []
        self.leads = []
        self.fs_mass = []
        self.title = ''
        self.filee = ''
        self.metadata = None
        self.units = []

        self.event_annotations = []

    @staticmethod
    def t_vectors(signals, fs) -> list[list[float]]:
        t_vectors_mass = []
        for signal, total_fs in zip(signals, fs):
            sig_units_num = len(signal)
            siglen_s = sig_units_num / total_fs
            t_vector = list(np.linspace(0, siglen_s, sig_units_num))
            t_vectors_mass.append(t_vector)
        return t_vectors_mass

    def select_events(self):
        def set_rect(ax_examp, row=None, col=None):
            if row != None and col != None:
                init_xlim = self.axs[row, col].get_xlim()
                init_ylim = self.axs[row, col].get_ylim()
                patch = self.axs[row, col].add_patch(Rectangle((init_xlim[0], init_ylim[0]),
                                                               init_xlim[1] - init_xlim[0],
                                                               init_ylim[1] - init_ylim[0],
                                                               edgecolor='red',
                                                               facecolor='none',
                                                               lw=4))

            elif row == None and col != None:
                init_xlim = self.axs[col].get_xlim()
                init_ylim = self.axs[col].get_ylim()
                patch = self.axs[col].add_patch(Rectangle((init_xlim[0], init_ylim[0]),
                                                          init_xlim[1] - init_xlim[0],
                                                          init_ylim[1] - init_ylim[0],
                                                          edgecolor='red',
                                                          facecolor='none',
                                                          lw=4))
            elif row == None and col == None:
                init_xlim = self.axs.get_xlim()
                init_ylim = self.axs.get_ylim()
                patch = self.axs.add_patch(Rectangle((init_xlim[0], init_ylim[0]),
                                                          init_xlim[1] - init_xlim[0],
                                                          init_ylim[1] - init_ylim[0],
                                                          edgecolor='red',
                                                          facecolor='none',
                                                          lw=4))
            else:
                patch = None

            return ax_examp, patch

        def set_span():
            self.span.set_visible(False)
            self.span = SpanSelector(
                self.selected_ax,
                onselect,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.5, facecolor="tab:blue"),
                interactive=True,
                drag_from_anywhere=True
            )

        def onclick(event):
            event_ax_code = str(event).split(' ')[-1].replace('inaxes=', '')
            try:
                codes_part = str(self.ax_codes[event_ax_code][1]).split('_')
            except KeyError:
                codes_part = None
            if event_ax_code == "None":
                pass
            elif not codes_part:
                pass
            elif str(self.selected_ax) == event_ax_code:
                pass
            else:
                self.patch.remove()
                if self.ax_type == 1:
                    self.axs, self.patch = set_rect(self.axs)
                    self.selected_ax = self.axs
                    set_span()
                elif self.ax_type == 2:
                    self.axs, self.patch = set_rect(self.axs, col=int(codes_part[-1]))
                    self.selected_ax = self.axs[int(codes_part[-1])]
                    set_span()
                else:
                    self.axs, self.patch = set_rect(self.axs, row=int(codes_part[-2]), col=int(codes_part[-1]))
                    self.selected_ax = self.axs[int(codes_part[-2]), int(codes_part[-1])]
                    set_span()
                plt.show()

        def init_selector():
            if self.ax_type == 1:
                self.axs, self.patch = set_rect(self.axs)
                self.selected_ax = self.axs
                self.span = SpanSelector(
                    self.selected_ax,
                    onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.5, facecolor="tab:blue"),
                    interactive=True,
                    drag_from_anywhere=True
                )

            elif self.ax_type == 2:
                self.axs, self.patch = set_rect(self.axs, col=0)
                self.selected_ax = self.axs[0]
                self.span = SpanSelector(
                    self.selected_ax,
                    onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.5, facecolor="tab:blue"),
                    interactive=True,
                    drag_from_anywhere=True
                )

            else:
                self.axs, self.patch = set_rect(self.axs, row=0, col=0)
                self.selected_ax = self.axs[0, 0]
                self.span = SpanSelector(
                    self.selected_ax,
                    onselect,
                    "horizontal",
                    useblit=True,
                    props=dict(alpha=0.5, facecolor="tab:blue"),
                    interactive=True,
                    drag_from_anywhere=True
                )

        def onselect(xmin, xmax):
            self.selected_range = (xmin, xmax)

        def add_event_on_graph(eventt):
            sig_num = eventt['sig_num']
            event_type = eventt['type']
            graph_position = self.position_sig_dict[sig_num]
            if self.ax_type == 3:
                r = int(graph_position.split('_')[-2])
                c = int(graph_position.split('_')[-1])
                init_xlim = self.axs[r, c].get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs[r, c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                         color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs[r, c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)

                self.axs[r, c].set_xlim(init_xlim[0], init_xlim[1])
            elif self.ax_type == 2:
                c = int(graph_position.split('_')[-1])
                init_xlim = self.axs[c].get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs[c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                      color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs[c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)
                self.axs[c].set_xlim(init_xlim[0], init_xlim[1])
            else:
                init_xlim = self.axs.get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs.axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                      color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs.axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)
                self.axs.set_xlim(init_xlim[0], init_xlim[1])

        def add_events():
            for select_span in self.spans_list:
                select_span.set_visible(False)
            for eventt in self.event_annotations:
                add_event_on_graph(eventt)
            # print(self.event_annotations)

        def button_LVP_handler(event):
            selected_ax = self.ax_codes[str(self.selected_ax)]
            rangee = self.selected_range
            sig_num = selected_ax[0]
            event = {'sig_num': sig_num, 'range': rangee, 'type': 'LVP'}
            self.event_annotations.append(event)
            # print(self.event_annotations)
            add_events()
            plt.show()

        def button_LAP_handler(event):
            selected_ax = self.ax_codes[str(self.selected_ax)]
            rangee = self.selected_range
            sig_num = selected_ax[0]
            event = {'sig_num': sig_num, 'range': rangee, 'type': 'LAP'}
            self.event_annotations.append(event)
            add_events()
            plt.show()

        def button_UNDO_handler(event):
            if self.event_annotations:
                self.event_annotations.pop()
                add_events()
                plt.show()
            else:
                pass

        def button_SAVE_handler(event):
            print(self.event_annotations)
            plt.close('all')

        def init_ui():
            tvectors = UIselector.t_vectors(self.signals, self.fs_mass)
            n_cols = 3
            if len(self.signals) < n_cols:
                n_cols = 1
            expected_n_rows = int(np.ceil(len(self.signals) / n_cols))
            plt.rc('font', size=6)
            self.fig, self.axs = plt.subplots(nrows=expected_n_rows, ncols=n_cols, figsize=(14, 7))
            self.fig.canvas.manager.set_window_title(self.title)
            plt.subplots_adjust(left=0.027, bottom=0.147, right=0.98, top=0.98, wspace=0.082, hspace=0.133)
            counter_rows = 0
            counter_cols = 0
            signal_counter = 0

            self.ax_codes = {}
            self.ax_type = None
            for signal, lead, unit, tvector in zip(self.signals, self.leads, self.units, tvectors):
                if n_cols == 1 and expected_n_rows == 1:
                    self.ax_type = 1
                    plt.plot(tvector, signal, linewidth=0.8)
                    plt.legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    plt.xlabel('[Sec]')
                    self.ax_codes[str(self.axs)] = [signal_counter, None]
                elif n_cols < 2 or expected_n_rows < 2:
                    self.ax_type = 2
                    self.axs[counter_rows].plot(tvector, signal, linewidth=0.8)
                    self.axs[counter_rows].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    self.axs[counter_rows].set_xlabel('[Sec]')
                    self.ax_codes[str(self.axs[counter_rows])] = [signal_counter, 'ax_' + str(counter_rows)]
                else:
                    self.ax_type = 3
                    if counter_rows > (expected_n_rows - 1):
                        counter_cols = counter_cols + 1
                        counter_rows = 0
                    self.axs[counter_rows, counter_cols].plot(tvector, signal, linewidth=0.8)  # , color= '#A40483')
                    self.axs[counter_rows, counter_cols].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    if counter_rows == (expected_n_rows - 1):
                        self.axs[counter_rows, counter_cols].set_xlabel('[Sec]')
                    self.ax_codes[str(self.axs[counter_rows, counter_cols])] = [signal_counter,
                                                                                'ax_' + str(counter_rows) + '_' + str(
                                                                                    counter_cols)]

                counter_rows = counter_rows + 1
                signal_counter = signal_counter + 1

                self.position_sig_dict = {}
                positions = self.ax_codes.values()
                for sigpos in positions:
                    self.position_sig_dict[sigpos[0]] = sigpos[1]

            self.LVP = Button(pylab.axes([0.005, 0.005, 0.1, 0.09]), 'LVP', color='greenyellow')
            self.LAP = Button(pylab.axes([0.115, 0.005, 0.1, 0.09]), 'LAP', color='khaki')
            self.UNDO = Button(pylab.axes([0.225, 0.005, 0.1, 0.09]), 'UNDO')
            self.SAVE = Button(pylab.axes([0.855, 0.005, 0.1, 0.09]), 'SAVE')

        init_ui()
        init_selector()
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
        add_events()
        self.LVP.on_clicked(button_LVP_handler)
        self.LAP.on_clicked(button_LAP_handler)
        self.UNDO.on_clicked(button_UNDO_handler)
        self.SAVE.on_clicked(button_SAVE_handler)
        plt.show(block=True)
        # plt.close(self.fig)

        return self.event_annotations


    def create_img_with_events(self, path_to_save):
        def add_event_on_graph(eventt):
            sig_num = eventt['sig_num']
            event_type = eventt['type']
            graph_position = self.position_sig_dict[sig_num]
            if self.ax_type == 3:
                r = int(graph_position.split('_')[-2])
                c = int(graph_position.split('_')[-1])
                init_xlim = self.axs[r, c].get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs[r, c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                         color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs[r, c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)

                self.axs[r, c].set_xlim(init_xlim[0], init_xlim[1])
            elif self.ax_type == 2:
                c = int(graph_position.split('_')[-1])
                init_xlim = self.axs[c].get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs[c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                      color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs[c].axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)
                self.axs[c].set_xlim(init_xlim[0], init_xlim[1])
            else:
                init_xlim = self.axs.get_xlim()

                if event_type == 'LVP':
                    select_span = self.axs.axvspan(eventt['range'][0], eventt['range'][1], alpha=1,
                                                      color='greenyellow')
                    self.spans_list.append(select_span)
                if event_type == 'LAP':
                    select_span = self.axs.axvspan(eventt['range'][0], eventt['range'][1], alpha=1, color='khaki')
                    self.spans_list.append(select_span)
                self.axs.set_xlim(init_xlim[0], init_xlim[1])

        def add_events():
            for select_span in self.spans_list:
                select_span.set_visible(False)
            for eventt in self.event_annotations:
                add_event_on_graph(eventt)
            # print(self.event_annotations)

        def init_ui():
            tvectors = UIselector.t_vectors(self.signals, self.fs_mass)
            n_cols = 3
            if len(self.signals) < n_cols:
                n_cols = 1
            expected_n_rows = int(np.ceil(len(self.signals) / n_cols))
            plt.rc('font', size=6)
            self.fig, self.axs = plt.subplots(nrows=expected_n_rows, ncols=n_cols, figsize=(14, 9.5))
            self.fig.canvas.manager.set_window_title(self.title)
            plt.subplots_adjust(left=0.027, bottom=0.06, right=0.98, top=0.98, wspace=0.082, hspace=0.185)
            counter_rows = 0
            counter_cols = 0
            signal_counter = 0

            self.ax_codes = {}
            self.ax_type = None
            for signal, lead, unit, tvector in zip(self.signals, self.leads, self.units, tvectors):
                if n_cols == 1 and expected_n_rows == 1:
                    self.ax_type = 1
                    plt.plot(tvector, signal, linewidth=0.8)
                    plt.legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    plt.xlabel('[Sec]')
                    self.ax_codes[str(self.axs)] = [signal_counter, None]
                elif n_cols < 2 or expected_n_rows < 2:
                    self.ax_type = 2
                    self.axs[counter_rows].plot(tvector, signal, linewidth=0.8)
                    self.axs[counter_rows].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    self.axs[counter_rows].set_xlabel('[Sec]')
                    self.ax_codes[str(self.axs[counter_rows])] = [signal_counter, 'ax_' + str(counter_rows)]
                else:
                    self.ax_type = 3
                    if counter_rows > (expected_n_rows - 1):
                        counter_cols = counter_cols + 1
                        counter_rows = 0
                    self.axs[counter_rows, counter_cols].plot(tvector, signal, linewidth=0.8)  # , color= '#A40483')
                    self.axs[counter_rows, counter_cols].legend([str(lead) + '[' + str(unit) + ']'], loc='upper right')
                    if counter_rows == (expected_n_rows - 1):
                        self.axs[counter_rows, counter_cols].set_xlabel('[Sec]')
                    self.ax_codes[str(self.axs[counter_rows, counter_cols])] = [signal_counter,
                                                                                'ax_' + str(counter_rows) + '_' + str(
                                                                                    counter_cols)]

                counter_rows = counter_rows + 1
                signal_counter = signal_counter + 1

                self.position_sig_dict = {}
                positions = self.ax_codes.values()
                for sigpos in positions:
                    self.position_sig_dict[sigpos[0]] = sigpos[1]

        init_ui()
        add_events()
        plt.savefig(path_to_save, dpi=500)
        plt.close('all')
        return path_to_save
