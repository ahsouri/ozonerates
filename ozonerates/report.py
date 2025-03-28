import os.path
import os
import glob
import time
import numpy as np
from fpdf import FPDF
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from mpl_toolkits.basemap import Basemap
from ozonerates.tools import error_averager


def plotter(X, Y, Z, fname: str, title: str, unit: int, vmin, vmax):

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes()
    map = Basemap(projection='cyl', llcrnrlat=np.min(Y.flatten()), urcrnrlat=np.max(Y.flatten()),
                  llcrnrlon=np.min(X.flatten()), urcrnrlon=np.max(X.flatten()), resolution="i")
    im = ax.imshow(Z, origin='lower',
                   extent=[np.min(X.flatten()), np.max(X.flatten()),
                           np.min(Y.flatten()), np.max(Y.flatten())],
                   interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax,
                   cmap=mpl.colormaps['rainbow'])
    map.drawcoastlines()
    map.drawcountries()
    x_ticks = np.arange(np.min(X.flatten()),
                        np.max(X.flatten()), 40)
    x_labels = np.linspace(np.min(X.flatten()), np.max(
        X.flatten()), np.size(x_ticks))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    y_ticks = np.arange(np.min(Y.flatten()), np.max(Y.flatten()), 20)
    y_labels = np.linspace(np.min(Y.flatten()), np.max(
        Y.flatten()), np.size(y_ticks))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=18)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plotting lat and lon
    plt.xlabel('Lon', fontsize=20)
    plt.ylabel('Lat', fontsize=20)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=18)
    if unit == 1:
        cbar.set_label(r'$[ \times 10^{15}molec.cm^{-2}] $', fontsize=18)
    elif unit == 2:
        cbar.set_label('$ [Unitless] $', fontsize=18)
    elif unit == 3:
        cbar.set_label('$ [ppbv/hr] $', fontsize=18)
    elif unit == 4:
        cbar.set_label(r'$[ \times ppbv/10^{15}molec.cm^{-2}] $', fontsize=18)
    elif unit == 5:
        cbar.set_label(r'$[ppbv] $', fontsize=18)
    elif unit == 6:
        cbar.set_label(r'$[\times 10^{-3}s^{-1}] $', fontsize=18)
    elif unit == 7:
        cbar.set_label(r'$[\times 10^{-6}s^{-1}] $', fontsize=18)
    elif unit == 8:
        cbar.set_label(r'$[\times 10^{18}molec.cm^{-3}] $', fontsize=18)
    plt.title(title, loc='left', fontweight='bold', fontsize=20)
    plt.tight_layout()
    fig.savefig(fname, format='png', dpi=300)
    plt.close()


def topdf(fname: str, folder: str, pdf_output: str):
    ''' 
    save all pngs to a pdf report
    '''
    def header(pdfobj, title, fsize=22):
        # Arial bold 15
        pdfobj.set_font('Arial', 'B', fsize)
        # Calculate width of title and position
        w = pdfobj.get_string_width(title) + 6
        pdfobj.set_x((210 - w) / 2)
        pdfobj.set_fill_color(255, 255, 255)
        pdfobj.set_text_color(0, 0, 0)
        # Thickness of frame (1 mm)
        pdfobj.set_line_width(1)
        # Title
        pdfobj.cell(w, 9, title, 1, 1, 'C', 1)
        # Line break
        pdfobj.ln(10)
        return w

    def body(pdfobj, bd1):
        # Times 12
        pdfobj.set_font('Times', '', 12)
        # Output justified text
        pdfobj.multi_cell(0, 5, bd1)
        # Line break
        pdfobj.ln(1)

    # call the fpdf obj
    pdf = FPDF(orientation="landscape")
    pdf.add_page()
    title = 'The ozonerates report generated by the OzoneRates tool'
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(280, 100, txt=title, border=0, ln=1, align="C")
    pdf.cell(280, 20, txt='Amir H. Souri', border=0, ln=1, align="C")
    pdf.cell(280, 20, txt='Contact: ahsouri@gmail.com',
             border=0, ln=1, align="C")
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    pdf.cell(280, 20, txt='Created at: ' +
             current_time, border=0, ln=1, align="C")

    # printing grayscales
    map_plots = sorted(glob.glob('temp/*' + fname + '*.png'))
    for fname in map_plots:
        pdf.add_page()
        pdf.image(fname, h=pdf.eph, w=pdf.epw+20)

    # writing
    if not os.path.exists(folder):
        os.makedirs(folder)

    pdf.output(folder + '/' + pdf_output, 'F')


def report(data, fname: str, folder: str):
    '''
    '''
    if not os.path.exists('temp'):
        os.makedirs('temp')

    plotter(data.longitude, data.latitude, np.nanmean(data.vcd_no2, axis=0).squeeze(), 'temp/A_vcd_no2_' +
            fname + '.png', 'VCD NO2', 1, 0, 5)
    plotter(data.longitude, data.latitude,  np.nanmean(data.vcd_hcho, axis=0).squeeze(), 'temp/D_vcd_form_' +
            fname + '.png', 'VCD FORM', 1, 0, 25)
    plotter(data.longitude, data.latitude, np.nanmean(data.hcho_vmr, axis=0).squeeze(), 'temp/F_vcd_form_vmr_' +
            fname + '.png', 'HCHO vmr (PBL)', 5, 0, 5)
    plotter(data.longitude, data.latitude, np.nanmean(data.no2_vmr, axis=0).squeeze(), 'temp/C_vcd_no2_vmr_' +
            fname + '.png', 'NO2 vmr (PBL)', 5, 0, 2)
    plotter(data.longitude, data.latitude, np.nanmean(data.vcd_no2_factor, axis=0).squeeze(), 'temp/B_vcd_no2_factor_' +
            fname + '.png', 'NO2 factor (PBL)', 4, 0, 0.3)
    plotter(data.longitude, data.latitude, np.nanmean(data.vcd_hcho_factor, axis=0).squeeze(), 'temp/E_vcd_hcho_factor_' +
            fname + '.png', 'FORM factor (PBL)', 4, 0, 0.3)
    plotter(data.longitude, data.latitude, np.nanmean(data.FNR, axis=0).squeeze(), 'temp/G_fnr_' +
            fname + '.png', 'FNR (PBL)', 2, 0, 10)
    plotter(data.longitude, data.latitude, np.nanmean(data.jo1d, axis=0).squeeze(), 'temp/H_jo1d_' +
            fname + '.png', 'JO1D', 7, 0, 100)
    plotter(data.longitude, data.latitude, np.nanmean(data.jno2, axis=0).squeeze(), 'temp/I_jno2_' +
            fname + '.png', 'JNO2', 6, 0, 20)
    plotter(data.longitude, data.latitude, np.nanmean(data.H2O, axis=0).squeeze(), 'temp/IA_H2O_' +
            fname + '.png', 'H2O', 8, 0, 1)
    plotter(data.longitude, data.latitude, np.nanmean(data.jno2_contrib, axis=0).squeeze(), 'temp/K_jno2_contrib_' +
            fname + '.png', 'JNO2 contribution to PO3', 3, -1, 5)
    plotter(data.longitude, data.latitude, np.nanmean(data.jo1d_contrib, axis=0).squeeze(), 'temp/J_jo1d_contrib_' +
            fname + '.png', 'JO1D contribution to PO3', 3, -1, 5)
    plotter(data.longitude, data.latitude, np.nanmean(data.hcho_vmr_contrib, axis=0).squeeze(), 'temp/L_hcho_contrib_' +
            fname + '.png', 'FORM contribution to PO3', 3, -1, 5)
    plotter(data.longitude, data.latitude, np.nanmean(data.no2_vmr_contrib, axis=0).squeeze(), 'temp/M_no2_contrib_' +
            fname + '.png', 'NO2 contribution to PO3', 3, -1, 5)
    plotter(data.longitude, data.latitude, np.nanmean(data.h2o_contrib, axis=0).squeeze(), 'temp/MA_h2o_contrib_' +
            fname + '.png', 'H2O contribution to PO3', 3, -1, 1)
    plotter(data.longitude, data.latitude, np.nanmean(data.PO3, axis=0).squeeze(), 'temp/N_po3_' +
            fname + '.png', 'PO3', 3, -1, 10)
    po3_error_rand = error_averager(data.po3_err_rand**2)
    po3_error_sys = np.sqrt(np.nanmean(data.po3_err_sys**2, axis=0)).squeeze()
    plotter(data.longitude, data.latitude, po3_error_rand, 'temp/O_po3_err_rand' +
            fname + '.png', 'PO3_err_rand', 3, 0, 3)
    plotter(data.longitude, data.latitude, po3_error_sys, 'temp/P_po3_err_sys' +
            fname + '.png', 'PO3_err_sys', 3, 0, 3)
    plotter(data.longitude, data.latitude, np.sqrt(po3_error_rand**2+po3_error_sys**2), 'temp/Q_po3_err_tot_' +
            fname + '.png', 'PO3_err_tot', 3, 0, 3.0)
    plotter(data.longitude, data.latitude, np.abs(100.0*np.sqrt(po3_error_rand**2+po3_error_sys**2)/np.nanmean(data.PO3, axis=0).squeeze()), 'temp/R_po3_err_tot_rel_' +
            fname + '.png', 'PO3_err_tot_rel', 2, 0, 100.0)
    topdf(fname, folder, 'PO3_report_' + fname + '.pdf')
