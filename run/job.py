import yaml
from ozonerates import ozonerates
from pathlib import Path
import sys

# Read the control file
with open('./control.yml', 'r') as stream:
    try:
        ctrl_opts = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

ctm_dir = ctrl_opts['ctm_dir']
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
ozonerates_obj.read_data('GMI', Path(ctm_dir),
                             sat_path, str(year) + f"{month:02}",output_folder = output_nc_inputparam_dir,
                             read_ak=False, trop=True, num_job=num_job)
ozonerates_obj.po3estimate_empirical(
           output_nc_inputparam_dir, output_nc_inputparam_dir, str(year) + '-' + f"{month:02}" +
           '-01', str(year) + '-' + f"{month+1:02}" + '-01')
ozonerates_obj.reporting("PO3_estimates_" + str(year) + f"{month:02}",folder=output_pdf_dir)