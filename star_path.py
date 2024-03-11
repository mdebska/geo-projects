import numpy as np
import matplotlib.pyplot as plt


def wykres2d_warsaw(dekl, rekt):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks(range(0, 90 + 10, 10))
    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    ax.set_yticklabels(yLabel)
    ax.set_rlim(0, 90)
    szer1 = 52  # stopnie szerokości geograficznej
    dl1 = 21  # stopnie długości geograficznej
    rekt_s = hms2rad([6, 37, 43.973])
    dekl_s = dms2rad([23, 4, 8.89])
    roznica = []
    h = ['', '', '', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, '', '', '']
    hst = ['', '', '', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, '', '', '', '', '']
    for i in range(0, 24):
        jd = julday(2023, 7, 1, i)
        g = GMST(jd)
        lst = dl1 + g * 15
        # dla gwiazdy
        t1 = np.deg2rad(lst) - rekt
        h1 = np.arcsin(
            np.sin(np.deg2rad(szer1)) * np.sin(dekl) + np.cos(np.deg2rad(szer1)) * np.cos(dekl) * np.cos(t1))
        az = np.arctan2(-(np.cos(dekl) * np.sin(t1)),
                        (np.cos(np.deg2rad(szer1)) * np.sin(dekl) - np.sin(np.deg2rad(szer1)) * np.cos(
                            dekl) * np.cos(t1)))
        ax.scatter(az, 90 - np.rad2deg(h1), color='red')
        ax.text(az+0.05, 85 - np.rad2deg(h1), h[i], horizontalalignment='left', verticalalignment='center_baseline')
        # dla słońca
        ts = np.deg2rad(lst) - rekt_s
        hs = np.arcsin(
            np.sin(np.deg2rad(szer1)) * np.sin(dekl_s) + np.cos(np.deg2rad(szer1)) * np.cos(dekl_s) * np.cos(ts))
        azs = np.arctan2(-(np.cos(dekl_s) * np.sin(ts)),
                        (np.cos(np.deg2rad(szer1)) * np.sin(dekl_s) - np.sin(np.deg2rad(szer1)) * np.cos(
                            dekl_s) * np.cos(ts)))
        ax.scatter(azs, 90 - np.rad2deg(hs), color='orange')
        ax.text(azs + 0.05, 92 - np.rad2deg(hs), hst[i], horizontalalignment='right', verticalalignment='top')
        roznica.append(h[i])
        roznica.append(dms2deg(rad2dms(abs(az - azs))))
    print(roznica)
    plt.show()


def wykres2d_equator(dekl, rekt):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks(range(0, 90 + 10, 10))
    yLabel = ['90', '', '', '', '50', '', '30', '', '10', '']
    ax.set_yticklabels(yLabel)
    ax.set_rlim(0, 90)
    h = [2, 3, '', '', '', '', 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, '', '', '', '', '', 1]
    hst = ['', '', '', '', '', 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, '', '', '', '', '', '', 1]
    szer1 = 0  # stopnie szerokości geograficznej
    dl1 = 21  # stopnie długości geograficznej
    rekt_s = hms2rad([6, 37, 43.973])
    dekl_s = dms2rad([23, 4, 8.89])
    roznica = []
    for i in range(0, 24):
        jd = julday(2023, 7, 1, i)
        g = GMST(jd)
        lst = dl1 + g * 15
        t1 = np.deg2rad(lst) - rekt
        h1 = np.arcsin(
            np.sin(np.deg2rad(szer1)) * np.sin(dekl) + np.cos(np.deg2rad(szer1)) * np.cos(dekl) * np.cos(t1))
        az = np.arctan2(-(np.cos(dekl) * np.sin(t1)),
                        (np.cos(np.deg2rad(szer1)) * np.sin(dekl) - np.sin(np.deg2rad(szer1)) * np.cos(
                            dekl) * np.cos(t1)))
        ax.scatter(az, 90 - np.rad2deg(h1), color='red')
        ax.text(az + 0.07, 90 - np.rad2deg(h1), h[i], horizontalalignment='left', verticalalignment='bottom')
        # dla słońca
        ts = np.deg2rad(lst) - rekt_s
        hs = np.arcsin(
            np.sin(np.deg2rad(szer1)) * np.sin(dekl_s) + np.cos(np.deg2rad(szer1)) * np.cos(dekl_s) * np.cos(ts))
        azs = np.arctan2(-(np.cos(dekl_s) * np.sin(ts)),
                         (np.cos(np.deg2rad(szer1)) * np.sin(dekl_s) - np.sin(np.deg2rad(szer1)) * np.cos(
                             dekl_s) * np.cos(ts)))
        ax.scatter(azs, 90 - np.rad2deg(hs), color='orange')
        ax.text(azs + 0.03, 87 - np.rad2deg(hs), hst[i], horizontalalignment='left', verticalalignment='top')
        roznica.append(h[i])
        roznica.append(dms2deg(rad2dms(abs(az - azs))))
    print(roznica)
    plt.show()


def wykres3d_warsaw(dekl_rad, rekt_rad):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    r = 1
    u, v = np.mgrid[0:(2 * np.pi + 0.1):0.1, 0:np.pi:0.1]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    z[z < 0] = 0
    ax.plot_surface(x, y, z, alpha=0.1)
    szer1 = 52  # stopnie szerokości geograficznej
    dl1 = 21  # stopnie długości geograficznej
    h = ['', '', '', 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, '', '', '']
    for i in range(0, 24):
        jd = julday(2023, 7, 1, i)
        g = GMST(jd)
        lst = dl1 + g * 15
        t1 = np.deg2rad(lst) - rekt_rad
        h1 = np.arcsin(np.sin(np.deg2rad(szer1)) * np.sin(dekl_rad) + np.cos(np.deg2rad(szer1)) * np.cos(dekl_rad) * np.cos(t1))
        az = np.arctan2(-(np.cos(dekl_rad) * np.sin(t1)),
                        (np.cos(np.deg2rad(szer1)) * np.sin(dekl_rad) - np.sin(np.deg2rad(szer1)) * np.cos(dekl_rad) * np.cos(t1)))
        gx = r * np.sin(az) * np.cos(h1)
        gy = r * np.cos(az) * np.cos(h1)
        gz = r * np.sin(h1)
        if h[i] != '':
            ax.plot3D(gx, gy, gz, 'p', color='r')
            ax.text(gx + 0.03, gy+0.07, gz, h[i], horizontalalignment='left', verticalalignment='center_baseline')
    plt.show()


def wykres3d_equator(dekl_rad, rekt_rad):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    r = 1
    u, v = np.mgrid[0:(2 * np.pi + 0.1):0.1, 0:np.pi:0.1]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    z[z < 0] = 0
    ax.plot_surface(x, y, z, alpha=0.1)
    szer2 = 0  # stopnie szerokości geograficznej
    dl2 = 21  # stopnie długosći geograficznej
    h = ['', '', '', '', '', '', 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, '', '', '', '', '', '']
    for i in range(0, 24):
        jd = julday(2023, 7, 1, i)
        g = GMST(jd)
        lst = dl2 + g * 15
        t1 = np.deg2rad(lst) - rekt_rad
        h1 = np.arcsin(np.sin(np.deg2rad(szer2)) * np.sin(dekl_rad) + np.cos(np.deg2rad(szer2)) * np.cos(dekl_rad) * np.cos(t1))
        az = np.arctan2(-(np.cos(dekl_rad) * np.sin(t1)),
                        (np.cos(np.deg2rad(szer2)) * np.sin(dekl_rad) - np.sin(np.deg2rad(szer2)) * np.cos(dekl_rad) * np.cos(t1)))
        gx = r * np.sin(az) * np.cos(h1)
        gy = r * np.cos(az) * np.cos(h1)
        gz = r * np.sin(h1)
        if h[i] != '':
            ax.plot3D(gx, gy, gz, 'p', color='r')
            ax.text(gx + 0.03, gy+0.07, gz, h[i], horizontalalignment='left', verticalalignment='center_baseline')

    plt.show()


def wys2az_warsaw():
    h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    wys = [-0.17428919832901937, -0.14769712360232687, -0.08561261804045714, 0.007325426491923466, 0.12530630902093387, 0.26239073431322857, 0.41299044969754284, 0.5717819486322563, 0.7330727437804431, 0.8893338497510836, 1.027973667523045, 1.1255558924988458, 1.1487249443328253, 1.086048778719322, 0.964982198845458, 0.8158097361777165, 0.6559883603009316, 0.4951156001860887, 0.3395784962772821, 0.19477370420572518, 0.06610763452532692, -0.04064132418690861, -0.11949821128993734, -0.16505600334702042]
    az = [0.048608612416959796, 0.2818249875622365, 0.506635053680376, 0.7191307441063836, 0.9191142901268113, 1.1096667591742562, 1.2965754061372468, 1.4885121471329432, 1.698642158071217, 1.9485432574686583, 2.27446421846628, 2.7226046799055372, -3.0058482716995925, -2.4908317943883467, -2.1060160723575274, -1.8221243864646088, -1.5947203415548326, -1.3953446729865275, -1.2071205556044118, -1.0193552622403919, -0.8248534772726198, -0.6190825814489587, -0.4004422420252723, -0.1709040440882102]
    h_etykiety = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1]
    fig, (graph1, graph2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=1)
    graph1.scatter(h, np.rad2deg(wys), color='black', s=10)
    graph1.set_title("Zależność wysokości od czasu")
    graph1.set_xlabel("Czas [h]")
    graph1.set_xticks(h)
    graph1.set_xticklabels(h_etykiety)
    graph1.set_ylabel("Wysokość gwiazdy [°]")
    graph2.scatter(h, np.rad2deg(az), color='black', s=10)
    graph2.set_title("Zależność azymutu od czasu")
    graph2.set_xlabel("Czas (h)")
    graph2.set_xticks(h)
    graph2.set_xticklabels(h_etykiety)
    graph2.set_ylabel("Azymut gwiazdy [°]")

    plt.show()


def wys2az_equator():
    h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    wys = [-1.0799123115372278, -0.9959641961201817, -0.8317972282764319, -0.6290249274153272, -0.4085248227796724, -0.17990431346762797, 0.051668733092737146, 0.282427886038874, 0.5082336415168595, 0.7224671509461873, 0.9116774323816166, 1.0464664949460645, 1.0789708012261106, 0.9915701697990233, 0.8256037953695192, 0.6220089704236356, 0.40112720579657996, 0.1723414428597569, -0.05926229728852996, -0.289931447755344, -0.5154846128424316, -0.7291677506690863, -0.9171809185333509, -1.0494204034441514]
    az = [0.10168717411381517, 0.5304814475883941, 0.8006106785246674, 0.9520954977307713, 1.0344208722939994, 1.073940929321565, 1.0819617720928785, 1.0606363308995732, 1.0041305873208042, 0.8954766588922681, 0.6985654487349138, 0.3575552823922863, -0.11766434881963567, -0.5418413350288117, -0.8071201576365149, -0.9556895517830916, -1.036301355911114, -1.0746701021317722, -1.081737457532211, -1.05939901685404, -1.0015311989034943, -0.89069064016332, -0.6899838741397732, -0.3435389185201093]
    h_etykiety = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1]
    fig, (graph1, graph2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=2)
    graph1.scatter(h, np.rad2deg(wys), color='black', s=10)
    graph1.set_title("Zależność wysokości od czasu")
    graph1.set_xlabel("Czas [h]")
    graph1.set_xticks(h)
    graph1.set_xticklabels(h_etykiety)
    graph1.set_ylabel("Wysokość gwiazdy [°]")
    graph2.scatter(h, np.rad2deg(az), color='black', s=10)
    graph2.set_title("Zależność azymutu od czasu")
    graph2.set_xlabel("Czas (h)")
    graph2.set_xticks(h)
    graph2.set_xticklabels(h_etykiety)
    graph2.set_ylabel("Azymut gwiazdy [°]")
    plt.show()


def julday(y, m, d, h):
    if m <= 2:
        y = y - 1
        m = m + 12
    jd = np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + d + h / 24 - 1537.5
    return jd


def GMST(jd):
    T = (jd - 2451545) / 36525
    g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000
    g = (g % 360) / 15
    return g


def rad2dms(rad):
    dd = np.rad2deg(rad)
    dd = dd
    deg = int(np.trunc(dd))
    mnt = int(np.trunc((dd - deg) * 60))
    sec = ((dd - deg) * 60 - mnt) * 60
    dms = [deg, abs(mnt), abs(sec)]
    return dms


def deg2dms(dd):
    deg = int(np.trunc(dd))
    mnt = int(np.trunc((dd - deg) * 60))
    sec = ((dd - deg) * 60 - mnt) * 60
    dms = [deg, abs(mnt), abs(sec)]
    return dms


def dms2rad(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]
    deg = d + m / 60 + s / 3600
    rad = np.deg2rad(deg)
    return rad


def dms2deg(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]

    deg = d + m / 60 + s / 3600
    return deg


def hms2rad(dms):
    d = dms[0]
    m = dms[1]
    s = dms[2]

    deg = d + m / 60 + s / 3600
    rad = np.deg2rad(deg * 15)
    return rad


if __name__ == '__main__':
    a = input('Dla ktorych wspolrzednych chcesz narysowac wykres? (warszawa/rownik) ')
    b = input('Jaki typ wykresu narysowac? (2d/3d/az2wys) ')
    rekt = [7, 46, 45.037] # rektascenzja
    dekl = [27, 58, 2.98]  # deklinacja
    rekt_rad = hms2rad(rekt)
    dekl_rad = dms2rad(dekl)

    if a == 'warszawa':
        if b == '3d':
            wykres3d_warsaw(dekl_rad, rekt_rad)
        elif b == '2d':
            wykres2d_warsaw(dekl_rad, rekt_rad)
        else:
            wys2az_warsaw()
    else:
        if b == '3d':
            wykres3d_equator(dekl_rad, rekt_rad)
        elif b == '2d':
            wykres2d_equator(dekl_rad, rekt_rad)
        else:
            wys2az_equator()
