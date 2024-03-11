import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


def haversine(fi1, fi2, lam1, lam2):
    R = 6371e3
    fi1 = np.deg2rad(fi1)
    fi2 = np.deg2rad(fi2)
    lam1 = np.deg2rad(lam1)
    lam2 = np.deg2rad(lam2)
    dlam = lam2 - lam1
    dfi = fi2 - fi1
    a = np.sin(dfi / 2) ** 2 + np.cos(fi1) * np.cos(fi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d / 1000


def wysk2odl():
    global spec, dane, wys
    odl = []
    for i in range(len(spec)):
        odl.append(spec[i][0])
    fig, graph1 = plt.subplots(1, 1)
    plt.subplots_adjust(hspace=2)
    for i in range(0, 33):
        graph1.scatter(odl[i], np.rad2deg(wys[i]), color='black', s=10)
    for i in range(33, len(odl)):
        if wys[i] >= 0:
            graph1.scatter(odl[i], np.rad2deg(wys[i]), color='black', s=10)
        else:
            graph1.scatter(odl[i], np.rad2deg(wys[i]), color='red', s=10)
    graph1.set_title("Zależność wysokości kątowej od odległości")
    graph1.set_xlabel("Odległość [kilometry]")
    graph1.set_ylabel("Wysokość [stopnie]")
    plt.show()


def wys2time():
    global spec, dane, wys
    wyssam = h_sam
    time = []
    for i in range(1, len(dane[0])):
        time.append((dane[0][i] - dane[0][0]) / 60)  # w minutach
    fig, graph1 = plt.subplots(1, 1)
    plt.subplots_adjust(hspace=2)
    for i in range(0, 33):
        graph1.scatter(time[i], wyssam[i], color='black', s=10)
    for i in range(33, len(wys)):
        if wys[i] >= 0:
            graph1.scatter(time[i], wyssam[i], color='black', s=10)
        else:
            graph1.scatter(time[i], wyssam[i], color='red', s=10)
    graph1.set_title("Zależność wysokości od czasu")
    graph1.set_xlabel("Czas [minuty]")
    graph1.set_ylabel("Wysokość [metry]")
    plt.show()


def pred2time():
    global spec, dane, wys
    time = []
    pred = dane[6]
    for i in range(1, len(dane[0])):
        time.append((dane[0][i] - dane[0][0]) / 60)  # w minutach
    fig, graph1 = plt.subplots(1, 1)
    plt.subplots_adjust(hspace=2)
    for i in range(0, 33):
        graph1.scatter(time[i], pred[i]*1.85166, color='black', s=10)
    for i in range(33, len(wys)):
        if wys[i] >= 0:
            graph1.scatter(time[i], pred[i]*1.85166, color='black', s=10)
        else:
            graph1.scatter(time[i], pred[i]*1.85166, color='red', s=10)
    graph1.set_title("Zależność prędkości od czasu")
    graph1.set_xlabel("Czas [minuty]")
    graph1.set_ylabel("Prędkość samolotu [km/h]")
    plt.show()


def mapa():
    global request, flh, spec, wys
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 1, 1, projection=request.crs)
    extent = [-10, 23, 45, 53]
    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)
    ax.set_global()
    ax.add_image(request, 5)
    ax.stock_img()
    ax.coastlines()
    for i in range(0, 33):
        ax.plot([flh[1][i], flh[1][i + 1]], [flh[0][i], flh[0][i + 1]], transform=ccrs.PlateCarree(), color='b')
    for i in range(33, len(wys) - 1):
        if wys[i] >= 0:
            ax.plot([flh[1][i], flh[1][i + 1]], [flh[0][i], flh[0][i + 1]], transform=ccrs.PlateCarree(), color='b')
        else:
            ax.plot([flh[1][i], flh[1][i + 1]], [flh[0][i], flh[0][i + 1]], transform=ccrs.PlateCarree(), color='r')
    ax.plot([flh[1][0], flh[1][-1]], [flh[0][0], flh[0][-1]], transform=ccrs.PlateCarree(), color='g')
    plt.show()


def az2odl():
    global spec
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    radial_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000]
    radial_labels = ['0', '500', '1000', '1500', '2000', '2500', '3000']
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_labels)
    ax.set_rlim(0, 3000)
    azimuths = [item[1] for item in spec]
    distances = [item[0] for item in spec]
    for i in range(0, 33):
        ax.scatter(azimuths[i], distances[i], color='black', s=3)
    for i in range(33, len(wys) - 1):
        if wys[i] >= 0:
            ax.scatter(azimuths[i], distances[i], color='black', s=3)
        else:
            ax.scatter(azimuths[i], distances[i], color='red', s=3)
    ax.set_rmax(3000)
    plt.show()


def read_flightradar(file):
    """
    Parameters
    ----------
    file : .csv file - format as downloaded from fligthradar24
        DESCRIPTION.
    Returns
    -------
    all_data : numpy array
        columns are:
            0 - Timestamp - ?
            1 - year
            2 - month
            3 - day
            4 - hour
            5 - minute
            6 - second
            7 - Latitude [degrees]
            8 - Longitude [degrees]
            9 - Altitude [feet]
            10 - Speed [?]
            11 - Direction [?]
    """
    with open(file, 'r') as f:
        it = 0
        size = []
        Timestamp = []
        date = []
        UTC = []
        Latitude = []
        Longitude = []
        Altitude = []
        Speed = []
        Direction = []
        all_data = []
        for linia in f:
            if linia[0:1] != 'T':
                splited_line = linia.split(',')
                size.append(len(splited_line))
                it += 1
                Timestamp.append(int(splited_line[0]))
                full_date = splited_line[1].split('T')
                date.append(list(map(int, full_date[0].split('-'))))
                UTC.append(list(map(int, full_date[1].split('Z')[0].split(':'))))
                Latitude.append(float(splited_line[3].split('"')[1]))
                Longitude.append(float(splited_line[4].split('"')[0]))
                Altitude.append(float(splited_line[5]))
                Speed.append(float(splited_line[6]))
                Direction.append(float(splited_line[7]))
        all_data.append(Timestamp)
        all_data.append(date)
        all_data.append(UTC)
        all_data.append(Latitude)
        all_data.append(Longitude)
        all_data.append(Altitude)
        all_data.append(Speed)
        all_data.append(Direction)
    return all_data


dane = read_flightradar('lot3.csv')
a = 6378137  # metry
e2 = 0.00669438002290

N_lot = a / np.sqrt(1 - e2 * np.sin(np.deg2rad(dane[3][0])) ** 2)  # metry / rady
lat_lot = dane[3][0]  # stopnie
long_lot = dane[4][0]  # stopnie
h_lot = dane[5][0] / 3.280840 + 104 + 31.4  # metry

lat_sam = []
long_sam = []
h_sam = []
N_sam = []

for i in range(1, len(dane[3])):
    lat_sam.append(dane[3][i])  # stopnie
    long_sam.append(dane[4][i])  # stopnie
    h_sam.append(dane[5][i] / 3.280840 + 104 + 31.4)  # metry
    n = a / (np.sqrt(1 - e2 * np.sin(np.deg2rad(dane[3][i])) ** 2))  # metry / rady
    N_sam.append(n)

x_lot = (N_lot + h_lot) * np.cos(np.deg2rad(lat_lot)) * np.cos(np.deg2rad(long_lot))
y_lot = (N_lot + h_lot) * np.cos(np.deg2rad(lat_lot)) * np.sin(np.deg2rad(long_lot))
z_lot = (N_lot * (1 - e2) + h_lot) * np.sin(np.deg2rad(lat_lot))

x_sam = []
y_sam = []
z_sam = []
for i in range(len(N_sam)):
    x = (N_sam[i] + h_sam[i]) * np.cos(np.deg2rad(lat_sam[i])) * np.cos(np.deg2rad(long_sam[i]))
    y = (N_sam[i] + h_sam[i]) * np.cos(np.deg2rad(lat_sam[i])) * np.sin(np.deg2rad(long_sam[i]))
    z = ((N_sam[i] * (1 - e2) + h_sam[i]) * np.sin(np.deg2rad(lat_sam[i])))
    x_sam.append(x)
    y_sam.append(y)
    z_sam.append(z)

wekt_lot = [x_lot, y_lot, z_lot]
wekt_sam = []
for i in range(len(x_sam)):
    wekt = [x_sam[i], y_sam[i], z_sam[i]]
    wekt_sam.append(wekt)

wekt_sl = []
for i in range(len(wekt_sam)):
    wekt = [wekt_sam[i][0] - wekt_lot[0], wekt_sam[i][1] - wekt_lot[1], wekt_sam[i][2] - wekt_lot[2]]
    wekt_sl.append(wekt)

wekt_u = [np.cos(np.deg2rad(lat_lot)) * np.cos(np.deg2rad(long_lot)),
          np.cos(np.deg2rad(lat_lot)) * np.sin(np.deg2rad(long_lot)),
          np.sin(np.deg2rad(lat_lot))]

wekt_n = [-np.sin(np.deg2rad(lat_lot)) * np.cos(np.deg2rad(long_lot)),
          -np.sin(np.deg2rad(lat_lot)) * np.sin(np.deg2rad(long_lot)),
          np.cos(np.deg2rad(lat_lot))]

wekt_e = [-np.sin(np.deg2rad(long_lot)),
          np.cos(np.deg2rad(long_lot)),
          0]

wekt_Rneu = np.array(
    [[wekt_n[0], wekt_e[0], wekt_u[0]], [wekt_n[1], wekt_e[1], wekt_u[1]], [wekt_n[2], wekt_e[2], wekt_u[2]]])
wekt_Rneu_t = np.transpose(wekt_Rneu)

wekt_slneu = []
spec = []
azymuty = []
wysokosci = []
odleglosci = []
for i in range(len(wekt_sl)):
    wekt = np.dot(wekt_Rneu_t, wekt_sl[i])
    wekt_slneu.append(wekt)
    n = wekt[0]
    e = wekt[1]
    u = wekt[2]
    s = (np.sqrt(n ** 2 + e ** 2 + u ** 2)) / 1000
    az = np.arctan2(e, n)
    h = np.arcsin(u / np.sqrt(n ** 2 + e ** 2 + u ** 2))
    z = 90 - h
    tab = [s, az, h]
    wysokosci.append(h)
    azymuty.append(az)
    odleglosci.append(s)
    spec.append(tab)
flh = [dane[3], dane[4], dane[5]]
request = cimgt.GoogleTiles()

wys = []
for i in range(len(spec)):
    wys.append(spec[i][2])

prosta = haversine(lat_lot, lat_sam[-1], long_lot, long_sam[-1])

# wys2time()
# wysk2odl()
# mapa()
# az2odl()
# pred2time()
