import gc
import os
import glob
import shutil
import logging
import pandas as pd
from itertools import chain
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathos.multiprocessing import ProcessingPool as Pool
from xgboost_ray import RayDMatrix, RayParams, train

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(format="%(thread)d %(asctime)s %(name)s:%(levelname)s:%(lineno)d:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.comm.tcp").setLevel(logging.ERROR)
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None

model_target = "my"
model_version = "v2"

model_input_cols = ['x1']
for i in range(64):
    layer_feature = ['layer1_mean_' + str(i), 'layer1_std_' + str(i)]
    model_input_cols.extend(layer_feature)
for i in range(128):
    layer_feature = ['layer2_mean_' + str(i), 'layer2_std_' + str(i)]
    model_input_cols.extend(layer_feature)

sel_cols = model_input_cols.copy()
sel_cols.append('y')

train_dir = "/data/"
df_features = pd.read_feather("/data/fw_site_features.ft")
study_name = "xgb_2022_11_v0"

best_params = {'colsample_bynode': 0.9,
        'max_depth': 19,
        'num_boost_round': 10,
        'num_parallel_tree': 21,
        'subsample': 0.7,
        'lambda': 1.14}

xgb_params = {
    'colsample_bynode': best_params['colsample_bynode'],
    'learning_rate': 1,
    'max_depth': best_params['max_depth'],
    'num_parallel_tree': best_params['num_parallel_tree'],
    'objective': 'reg:squarederror',
    'subsample': best_params['subsample'],
    'tree_method': 'gpu_hist',
    'lambda': best_params['lambda']
}
logger.debug("model params loaded")


def load_data(fn):
    global df_features
    df = None
    try:
        df = pd.read_feather(fn)
        df = df.set_index('sid').join(df_features.set_index('sid')).reset_index()
    except Exception as e:
        logger.debug("read file error:" + fn)
        logger.exception(e)
        return None
    return df


def train_one(start_day, start_hour="00", force=False):
    global model_target, train_dir, sel_cols
    model_path =  "models/" + model_target + "/"
    try:
        os.makedirs(model_path)
    except Exception as e:
        logger.warning(e)
    try:
        os.chmod(model_path, 0o777)
    except Exception as e:
        logger.warning(e)
    model_file = model_path + "xgb_" + start_day.strftime("%Y%m%d") + start_hour + "_" + model_version + ".model"
    error = False
    if os.path.exists(model_file) and not force:
        logger.warning("model exists " + model_file)
        return model_file
    range_days = 20
    train_dirs = []
    for hist_year in range(10):
        train_start = start_day - relativedelta(years=hist_year) - relativedelta(days=int(range_days/2))
        train_dir_list = [train_dir + (train_start + relativedelta(days=x)).strftime("%Y%m%d") + start_hour + "/" for x in range(range_days)]
        train_dirs.extend(train_dir_list)
    data_files = [glob.glob(one_dir + "*.ft") for one_dir in train_dir_list]
    all_data_files = list(chain(*data_files))
    all_data_files.sort()
    logger.debug(model_target + " load period files: " + str(len(all_data_files)))
    df_samples = None
    if len(all_data_files)>0:
        try:
            with Pool(48) as pool:
                list_df = pool.map(load_data, all_data_files)
            list_df = [df for df in list_df if df is not None]
            if len(list_df)==0:
                logger.warning("no train data for model " + str(start_day))
                return 0
            df_samples = pd.concat(list_df, ignore_index=True)
            df_samples.dropna(subset=['lat', 'lon', 'y_t2m'], inplace=True)
            df_samples = df_samples[sel_cols]
            df_samples.reset_index(drop=True, inplace=True)
        except Exception as e:
            logger.warning("error at train model for " + str(start_day))
            logger.exception(e)
            error = True
    else:
        logger.debug("no train data for model " + str(start_day))
    if df_samples is not None:
        # mostly 120w rows, 360 cols
        logger.debug("start train " + model_target + " model " + str(start_day) + " use " + str(len(df_samples)) + " train data, boost round " + str(best_params['num_boost_round']))
        train_set = RayDMatrix(df_samples[model_input_cols], df_samples['y_t2m'])
        try:
            booster = train(
                xgb_params,
                train_set,
                num_boost_round=best_params['num_boost_round'],
                evals=[(train_set, "train")],
                verbose_eval=True,
                ray_params=RayParams(
                    num_actors=4,
                    gpus_per_actor=1,
                    cpus_per_actor=8,   # Divide evenly across actors per machine
                )
            )
            booster.save_model(model_file)
            logger.debug("model saved: " + model_file)
        except Exception as e:
            error = True
            logger.warning("train exception for model " + start_day.strftime("%Y%m%d"))
            logger.exception(e)
    gc.collect()
    if error:
        return None
    return model_file


if __name__=="__main__":
    start_date = datetime.strptime("20221101", "%Y%m%d")
    for d in range(30):
        train_day =  start_date + timedelta(days=d)
        try:
            shutil.rmtree("/tmp/ray")
        except Exception as e:
            logger.debug(e)
        # import importlib
        # importlib.reload(xgboost_ray) # not work, ray still wait actor
        ret = train_one(train_day, force=True)
        logger.debug("finish train: " + str(train_day))
    logger.debug("all train finished")
