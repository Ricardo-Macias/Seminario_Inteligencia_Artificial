import cv2
import matplotlib.pyplot as plt
import numpy as np
#from google.colab.patches import cv2_imshow
from IPython import display

def Transformacion_Similitud(q, x):
    dx = q[0]
    dy = q[1]
    theta = q[2]
    s = q[3]

    xp1 = [s*np.cos(theta)*x[0]-s*np.sin(theta)*x[1]+dx]
    xp2 = [s*np.sin(theta)*x[0]+s*np.cos(theta)*x[1]+dy]
    xp = np.array([xp1, xp2])
    return xp

def Distancia_Euclidiana(xr, xp):
    e = np.sqrt((xr[0]-xp[0])**2 + (xr[1]-xp[1])**2)
    return e

def Generar_Resultado(q, img_dst, img_ref):
    dx = q[0]
    dy = q[1]
    theta = q[2]
    s = q[3]

    T = np.matrix([[s*np.cos(theta), -s*np.sin(theta), dx],
                   [s*np.sin(theta), s*np.cos(theta), dy],
                   [0, 0, 1]])
    Tp = T.T

    N, M, _ = img_ref.shape
    n, m, _ = img_dst.shape

    img_out = cv2.warpPerspective(img_dst, T, (M, N))

    # Para pegar una imagen sobre otra
    thresh = cv2.threshold(cv2.cvtColor(
        img_out, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    foreground = img_out.copy()
    background = img_ref.copy()
    alpha = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float) / 255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1 - alpha, background)
    outImage = cv2.add(foreground, background)
    outImage = outImage.astype(np.uint8)

    return outImage


img_ref = cv2.imread('ref_1.png')
H, W, _ = img_ref.shape

img_dst = cv2.imread('des.png')
h, w, _ = img_dst.shape

detect = cv2.QRCodeDetector()
_, points, _ = detect.detectAndDecode(img_ref)

xr0 = points[0, 0]
xr1 = points[0, 1]
xr2 = points[0, 2]
# xr3 = points[0, 3]

x0 = [0, 0]
x1 = [w, 0]
x2 = [w, h]

#---------------------------------------------------------------
# ----------> AQUI ELEGIR LA CONFIGURACIÓN ADECUADA <----------
#---------------------------------------------------------------

xl = np.array([-5,-5 ,-5 ,-5 ])
xu = np.array([5, 5, 5,5 ])

G = 100
mu = 20 #padres
l = 40 #hijos
D = 2

x = np.zeros((D, mu+l))
sigma = np.zeros((D, mu+l))
fitness = np.zeros(mu+l)

p_plot = np.zeros(G)

for i in range(mu):
    x[:, i] = xl + (xu - xl) * np.random.rand(D)
    # AQUÍ AGREGAR EL VECTOR SIGMA ADECUADO
    sigma[:, i] = [, , , ]

    q = x[:, i]

    # 1. Calcular los puntos xi transformados
    # Usar la función Transformación_Similitud
    xp0 = Transformacion_Similitud(q, x0)
    xp1 = Transformacion_Similitud(q, x1)
    xp2 = Transformacion_Similitud(q, x2)

    # 2. Calcular errores entre xri y xpi
    # Usar la función Distancia_Euclidiana
    e0 = Distancia_Euclidiana(xr0, xp0)
    e1 = Distancia_Euclidiana(xr1, xp1)
    e2 = Distancia_Euclidiana(xr2, xr2)

    fitness[i] =

for g in range(G):
   ## ------------ AQUÍ TU CÓDIGO ------------- ##

   ## ----------------------------------------- ##

    Idx = np.argsort(fitness)
    x = x[:, Idx]
    sigma = sigma[:, Idx]
    fitness = fitness[Idx]

    p_plot[g] = np.min(fitness)

q = x[:, 0]
display.display(plt.gcf())
display.clear_output(wait=True)
img = Generar_Resultado(q, img_dst, img_ref)
cv2_imshow(img)

plt.plot(p_plot)
