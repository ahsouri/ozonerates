import yaml
from ozonerates import ozonerates
from pathlib import Path
import sys
from time import asctime, gmtime, strftime

# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

ctm_dir = ctrl_opts['ctm_dir']
ctm_type = ctrl_opts['ctm_type']
ctm_freq = ctrl_opts['ctm_freq']
algorithm = ctrl_opts['algorithm']
sensor = ctrl_opts['sensor']
sat_dir_no2 = ctrl_opts['sat_dir_no2']
sat_dir_hcho = ctrl_opts['sat_dir_hcho']
output_pdf_dir = ctrl_opts['output_pdf_dir']
output_nc_inputparam_dir = ctrl_opts['output_nc_inputparam_dir']
output_nc_po3_dir = ctrl_opts['output_nc_PO3_dir']
num_job = ctrl_opts['num_job']

year = int(sys.argv[1])
month = int(sys.argv[2])

ozonerates_obj = ozonerates()
sat_path = []
sat_path.append(Path(sat_dir_no2))
sat_path.append(Path(sat_dir_hcho))
ozonerates_obj.read_data(ctm_type, ctm_freq, str(sensor), Path(ctm_dir),
                             sat_path, str(year) + f"{month:02}",output_folder = output_nc_inputparam_dir,
                             read_ak=False, trop=True, num_job=num_job)
if month != 12:
   if algorithm == 'DNN':
      ozonerates_obj.po3estimate_dnn(
           output_nc_inputparam_dir, output_nc_inputparam_dir, str(year) + '-' + f"{month:02}" +
           '-01', str(year) + '-' + f"{month+1:02}" + '-01', num_job=num_job)
   if algorithm == 'LASSO':
      print("running LASSO")
      ozonerates_obj.po3estimate_empirical(
           output_nc_inputparam_dir, output_nc_inputparam_dir, str(year) + '-' + f"{month:02}" +
           '-01', str(year) + '-' + f"{month+1:02}" + '-01', num_job=num_job)
else:
   if algorithm == 'DNN':
      ozonerates_obj.po3estimate_dnn(
           output_nc_inputparam_dir, output_nc_inputparam_dir, str(year) + '-' + f"{month:02}" +
           '-01', str(year+1) + '-01' + '-01', num_job=num_job)
   if algorithm == 'LASSO':
      ozonerates_obj.po3estimate_empirical(
           output_nc_inputparam_dir, output_nc_inputparam_dir, str(year) + '-' + f"{month:02}" +
           '-01', str(year+1) + '-01' + '-01', num_job=num_job)

ozonerates_obj.reporting("PO3_estimates_" + str(year) + f"{month:02}",folder=output_pdf_dir)
if sensor == 'OMI':
   output_nc_name = "PO3_Global__OMI_____" + str(year) + f"{month:02}" + "_" + strftime('%Y%m%dT%H%M%SZ', gmtime()) + ".nc"
else:
   output_nc_name = "PO3_Global__TROPOMI_" + str(year) + f"{month:02}" + "_" + strftime('%Y%m%dT%H%M%SZ', gmtime()) + ".nc"
ozonerates_obj.writenc(output_nc_name,folder=output_nc_po3_dir)
