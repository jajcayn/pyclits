"""
created on Sep 22, 2017

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
"""


from geofield import DataField
from datetime import date, datetime
from os.path import split
import numpy as np


def load_station_data(filename, start_date, end_date, anom, to_monthly = False, dataset = 'ECA-station', offset = 1):
    """
    Data loader for station data.
    """
    
    print("[%s] Loading station data..." % (str(datetime.now())))
    g = DataField()
    g.load_station_data(filename, dataset, print_prog = False, offset_in_file = offset)
    print("** loaded")
    g.select_date(start_date, end_date)
    if anom:
        print("** anomalising")
        g.anomalise()
    if to_monthly:
        g.get_monthly_data()
    day, month, year = g.extract_day_month_year()
    print("[%s] Data from %s loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), g.location, str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g



def load_NCEP_data_monthly(filename, varname, start_date, end_date, lats, lons, level, anom):
    """
    Data loader for monthly reanalyses data. 
    """

    print("[%s] Loading monthly NCEP/NCAR data..." % str(datetime.now()))
    g = DataField()
    g.load(filename, varname, dataset = 'NCEP', print_prog = False)
    print("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if level is not None:
        g.select_level(level)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))

    return g
    
    
    
def load_ERA_data_daily(filename, varname, start_date, end_date, lats, lons, anom, parts = 1, logger_function = None):
    """
    Data loader for daily ERA-40 / ERA-Interim data.
    If more than one file, filename should be all letters they have got in common without suffix.
    """
    
    if logger_function is None:
        def logger(msg):
            print("[%s] %s" % (str(datetime.now()), msg))
    else:
        logger = logger_function
        
    logger("Loading daily ERA-40 / ERA-Interim data...")
    
    # if in one file, just load it
    if parts == 1:
        path, name = split(filename)
        if path != '':
            path += '/'
            g = DataField(data_folder = path)
        else:
            g = DataField()
        g.load(name, varname, dataset = 'ERA', print_prog = False)
    
    # if in more files, find them all and load them
    else:
        fnames = []
        glist = []
        Ndays = 0
        path, name = split(filename)
        if path != '':
            path += '/'
        else:
            path = '../data'
        for root, _, files in os.walk(path):
            if root == path:
                for f in files:
                    if name in f:
                        fnames.append(f)
        if parts != len(fnames): 
            logger("Something went wrong since %d files matching your filename were found instead of %d." % (len(fnames), parts))
            raise Exception('Check your files and enter correct number of files you want to load.')
        for f in fnames:
            g = DataField(data_folder = path + '/')                
            g.load(f, varname, dataset = 'ERA', print_prog = False)
            Ndays += g.time.shape[0]
            glist.append(g)
            
        data = np.zeros((Ndays, len(glist[0].lats), len(glist[0].lons)))
        time = np.zeros((Ndays,))
        n = 0
        for g in glist:
            Ndays_i = len(g.time)
            data[n:Ndays_i + n, ...] = g.data
            time[n:Ndays_i + n] = g.time
            n += Ndays_i
        g = DataField(data = data, lons = glist[0].lons, lats = glist[0].lats, time = time)
        del glist
        
    if not np.all(np.unique(g.time) == g.time):
        logger('**WARNING: Some fields are overlapping, trying to fix this... (please note: experimental feature)')
        doubles = []
        for i in range(g.time.shape[0]):
            if np.where(g.time == g.time[i])[0].shape[0] == 1:
                # if there is one occurence of time value do nothing
                pass
            else:
                # remember the indices of other occurences
                doubles.append(np.where(g.time == g.time[i])[0][1:])
        logger("... found %d multiple values (according to the time field)..." % (len(doubles)/4))
        delete_mask = np.squeeze(np.array(doubles)) # mask with multiple indices
        # time
        g.time = np.delete(g.time, delete_mask)
        # data
        g.data = np.delete(g.data, delete_mask, axis = 0)
        
        
    logger("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    g.average_to_daily()
    if anom:
        logger("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()
    logger("ERA-40 / ERA-Interim data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1])) 
           
    return g



def load_ECA_D_data_daily(filename, varname, start_date, end_date, lats, lons, anom, logger_function = None):
    """
    Data loader for daily ECA&D reanalysis data.
    """

    if logger_function is None:
        def logger(msg):
            print("[%s] %s" % (str(datetime.now()), msg))
    else:
        logger = logger_function

    logger("Loading daily ECA&D data...")
    g = DataField()
    g.load(filename, varname, dataset = 'ECA-reanalysis', print_prog = False)
    logger("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if anom:
        logger("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    logger("ECA&D data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))

    return g

    
    
def load_NCEP_data_daily(filename, varname, start_date, end_date, lats, lons, level, anom):
    """
    Data loader for daily reanalyses data. Filename in form path/air.sig995.%d.nc
    """
    
    print("[%s] Loading daily NCEP/NCAR data..." % str(datetime.now()))
    start_year = start_date.year
    end_year = end_date.year - 1
    glist = []
    Ndays = 0
    path, name = split(filename)
    path += "/"
    
    for year in range(start_year, end_year+1):
        g = DataField(data_folder = path)
        fname = name % year
        g.load(fname, varname, dataset = 'NCEP', print_prog = False)
        Ndays += len(g.time)
        glist.append(g)
        
    data = np.zeros((Ndays, len(glist[0].lats), len(glist[0].lons)))
    time = np.zeros((Ndays,))
    n = 0
    for g in glist:
        Ndays_i = len(g.time)
        data[n:Ndays_i + n, ...] = g.data
        time[n:Ndays_i + n] = g.time
        n += Ndays_i
        
    g = DataField(data = data, lons = glist[0].lons, lats = glist[0].lats, time = time)
    del glist
    print("** loaded")
    g.select_date(start_date, end_date)
    g.select_lat_lon(lats, lons)
    if level is not None:
        g.select_level(level)
    if anom:
        print("** anomalising")
        g.anomalise()
    day, month, year = g.extract_day_month_year()

    print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           
    return g



def load_enso_index(fname, nino_type, start_date, end_date, anom = False):
    """
    Data loader for various ENSO indices.
    """

    from dateutil.relativedelta import relativedelta

    g = DataField()

    enso_raw = np.loadtxt(fname, skiprows = 1)
    y = int(enso_raw[0,0])
    start_date_ts = date(y, 1, 1)
    enso_raw = enso_raw[:, 1:]
    enso_raw = enso_raw.reshape(np.prod(enso_raw.shape))

    g.data = enso_raw.copy()
    time = np.zeros_like(enso_raw, dtype = np.int32)
    delta = relativedelta(months = +1)
    d = start_date_ts
    for i in range(time.shape[0]):
        time[i] = d.toordinal()
        d += delta

    g.time = time.copy()
    g.location = ('NINO%s SSTs' % nino_type)

    print("** loaded")
    g.select_date(start_date, end_date)
    if anom:
        print("** anomalising")
        g.anomalise()
    _, month, year = g.extract_day_month_year()

    print("[%s] Nino%s data loaded with shape %s. Date range is %d/%d - %d/%d inclusive." 
        % (str(datetime.now()), nino_type, str(g.data.shape), month[0], 
           year[0], month[-1], year[-1]))
           
    return g
    