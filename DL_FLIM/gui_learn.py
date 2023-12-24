import PySimpleGUI as sg
import os.path
import numpy as np
# ------------------------------- This is to include a matplotlib figure in a Tkinter canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from util.run_model import get_inten,run_model,get_inten
from util.phasor_function import phasor_function


def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)
#%%
file_list_column1 = [
    [
        sg.Text("Data Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER_DATA-"),
        sg.FolderBrowse(),
    ],
]
    
file_list_column2 = [
    [
        sg.Text("IRF Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FILE_IRF-"),
        sg.FileBrowse(),
    ],
]

file_list_column3 = [
    [
        sg.Text("Pretrained model File"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER_MODEL-"),
        sg.FileBrowse(),
    ],
]

file_list_column4 = [
    [
        sg.Text("Data Name"),
        sg.In(size=(25, 1), enable_events=True, key="-DATA_NAME-"),
        sg.FileBrowse(),
    ],
]

# ----- Full layout -----
#Define components, four folder areas.
layout = [
        [sg.Column(file_list_column1),sg.Column(file_list_column2)],
        [sg.Column(file_list_column4),sg.Column(file_list_column3)],
#Define buttons and texts
        [sg.B('Plot phasor'),
         sg.B('Plot FLIM IMAGE'),
         sg.B('Plot intensity IMAGE'),
         sg.Text('Thresh'), sg.InputText(size=(5,1),key='Threshold_value'),
         sg.Checkbox('REAL DATA', default=True, key='REAL_DATA')],
#Define plots events
        [sg.Canvas(key='control_phasor'),
         sg.Canvas(key='control_Tau_A'),
         sg.Canvas(key='control_Tau_I'),
         sg.Canvas(key='control_intensity')],
#Define plot areas
        [sg.T('phasor image:'),
         sg.Column(
                 layout=[
                         [sg.Canvas(key='phasor_cv',
                       # it's important that you set this size
                         size=(400, 250)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 0)
    ),
                
        sg.T('Intensity image:'),
        sg.Column(
                layout=[
                        [sg.Canvas(key='Intensity_cv',
                       # it's important that you set this size
                        size=(350, 350)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 10)
    )],   
        
        [
        sg.T('Tau_A image:'),
        sg.Column(
                layout=[
                        [sg.Canvas(key='Tau_A_cv',
                       # it's important that you set this size
                        size=(350, 350)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 0)
    ),
        sg.T('Tau_I image:'),
        sg.Column(
                layout=[
                        [sg.Canvas(key='Tau_I_cv',
                       # it's important that you set this size
                        size=(350, 350)
                       )]
        ],
        background_color='#DAE0E6',
        pad=(0, 10)
        )
    ],

]

#Define window sizes
window = sg.Window("AdderNet FLIM TOOL", layout,size=(1030, 930))

# Run the Event Loop
# Implement functions in this while loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    elif event == "-FOLDER_DATA-":
        folder = values["-FOLDER_DATA-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f)) or f.lower().endswith((".*"))
        ]
    elif event == 'REAL_DATA':
        real_data_en=values["REAL_DATA"]
        
    elif event == 'Threshold_value':
        Threshold_value=values["Threshold_value"]

    elif event == "-FILE_IRF-":
        folder = values["-FILE_IRF-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f_irf
            for f_irf in file_list
            if os.path.isfile(os.path.join(folder, f_irf)) and f_irf.lower().endswith((".mat"))
        ] 
        
    elif event == "-FOLDER_MODEL-":
        folder = values["-FOLDER_MODEL-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".pth"))
        ]
    elif event == "-DATA_NAME-":
        folder_data = values["-DATA_NAME-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder_data)
        except:
            file_list = []

        fnames = [
            f_data
            for f_data in file_list
            if os.path.isfile(os.path.join(folder_data, f_data)) and f_data.lower().endswith((".*"))
        ]
    elif event == 'Plot phasor':
        # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
        folder_data = values["-FOLDER_DATA-"]
        
        filename_data = os.path.join(
            values["-FOLDER_DATA-"], values["-DATA_NAME-"]
        )
        
        filename_irf = os.path.join(
            values["-FOLDER_DATA-"], values["-FILE_IRF-"]
        )
        plt.figure(1)
        fig1 = plt.gcf()
        DPI = fig1.get_dpi()
        # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
        fig1.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
        
        real_data_en = values["REAL_DATA"]
        Threshold = values["Threshold_value"]
        uo,vo,u_i,ui,vi=phasor_function(folder_data,filename_data,folder_data,filename_irf,real_data_en,Threshold)
        
        Xedge=np.arange(0,1,0.002)
        Yedge=np.arange(0,1,0.002)
        
        plt.hist2d(ui, vi, bins=(Xedge, Yedge), range=None, density=False, weights=None, cmin=None, cmax=None, cmap=plt.cm.nipy_spectral)
        plt.colorbar()
        x = np.arange(0,1.005,0.005)
        circle = np.sqrt(0.25 - (x - 0.5)**2)
        plt.plot(x,circle)
        plt.ylim(0,0.5)
        plt.xlim(0,1)
        # plt.tight_layout() 
        # ------------------------------- Instead of plt.show()
        draw_figure_w_toolbar(window['phasor_cv'].TKCanvas, fig1, window['control_phasor'].TKCanvas)
        
    elif event == 'Plot FLIM IMAGE':
        # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
        pretrained_model = os.path.join(
            values["-FOLDER_DATA-"], values["-FOLDER_MODEL-"]
        )
        
        folder_data = values["-FOLDER_DATA-"]
        filename_data = os.path.join(
            values["-FOLDER_DATA-"], values["-DATA_NAME-"]
        )
        real_data_en = values["REAL_DATA"]
        real_data_en = values["REAL_DATA"]
        tau_amp,tau_inten=run_model(folder_data,filename_data,pretrained_model,real_data_en,real_data_en)
        
        plt.figure(2)
        fig2 = plt.gcf()
        DPI = fig2.get_dpi()
        fig2.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
        
        tau_amp=np.transpose(tau_amp,[1,0])
        plt.imshow(tau_amp, cmap='gist_stern')#nipy_spectral, 
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        plt.clim(0, 3)
        plt.axis('off')
        # plt.tight_layout() 

        draw_figure_w_toolbar(window['Tau_A_cv'].TKCanvas, fig2, window['control_Tau_A'].TKCanvas)
        
        plt.figure(3)
        fig3 = plt.gcf()
        DPI = fig3.get_dpi()
        fig3.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
        
        tau_inten=np.transpose(tau_inten,[1,0])
        plt.imshow(tau_inten, cmap='gist_stern')#nipy_spectral, gist_ncar,gist_stern
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        plt.clim(0, 4)
        plt.axis('off')
        # plt.tight_layout()
        
        draw_figure_w_toolbar(window['Tau_I_cv'].TKCanvas, fig3, window['control_Tau_I'].TKCanvas)
    
    
    elif event == 'Plot intensity IMAGE':
        # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
        folder_data = values["-FOLDER_DATA-"]
        
        filename_data = os.path.join(
            values["-FOLDER_DATA-"], values["-DATA_NAME-"]
        )
        plt.figure(4)
        fig4 = plt.gcf()
        DPI = fig4.get_dpi()
        # ------------------------------- you have to play with this size to reduce the movement error when the mouse hovers over the figure, it's close to canvas size
        fig4.set_size_inches(404 * 2 / float(DPI), 404 / float(DPI))
        # -------------------------------
        real_data_en = values["REAL_DATA"]
        intensity=get_inten(folder_data,filename_data,real_data_en)
        intensity=np.transpose(intensity,[1,0])
        plt.imshow(intensity, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        # plt.tight_layout() 
        # ------------------------------- Instead of plt.show()
        draw_figure_w_toolbar(window['Intensity_cv'].TKCanvas, fig4, window['control_intensity'].TKCanvas)
        
        pass
window.close()