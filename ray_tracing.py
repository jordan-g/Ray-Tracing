import numpy as np
import matplotlib.pyplot as plt

# background color
BACKGROUND_COLOR = (255, 255, 255)

# viewport size
V_w = 1
V_h = 1
d   = 1

# canvas size
C_w = 512
C_h = 512

# scene
class Scene:
	def __init__(self):
		self.spheres = []
		self.lights  = []

scene = Scene()

class Sphere:
	def __init__(self, center, radius, color, specular):
		self.center   = center
		self.radius   = radius
		self.color    = color
		self.specular = specular

class Light:
	def __init__(self, type, intensity, position=None, direction=None):
		self.type      = type
		self.intensity = intensity
		self.position  = position
		self.direction = direction

def canvas_to_viewport(x, y):
	return np.array([x*V_w/C_w, y*V_h/C_h, d])

def trace_ray(O, D, t_min, t_max):
	closest_t = np.inf
	closest_sphere = None

	for sphere in scene.spheres:
		t_1, t_2 = intersect_ray_sphere(O, D, sphere)

		if t_min < t_1 < t_max and t_1 < closest_t:
			closest_t = t_1
			closest_sphere = sphere

		if t_min < t_2 < t_max and t_2 < closest_t:
			closest_t = t_2
			closest_sphere = sphere

	if closest_sphere is None:
		return BACKGROUND_COLOR

	P = O + closest_t*D # compute intersection
	N = P - closest_sphere.center
	N = N/np.linalg.norm(N)

	return closest_sphere.color*compute_lighting(P, N, -D, closest_sphere.specular)

def intersect_ray_sphere(O, D, sphere):
	C = sphere.center
	R = sphere.radius

	oc = O - C

	k_1 = np.dot(D, D)
	k_2 = 2*np.dot(oc, D)
	k_3 = np.dot(oc, oc) - R**2

	discriminant = k_2*k_2 - 4*k_1*k_3

	if discriminant < 0:
		return np.inf, np.inf

	t_1 = (-k_2 + np.sqrt(discriminant))/(2*k_1)
	t_2 = (-k_2 - np.sqrt(discriminant))/(2*k_1)

	return t_1, t_2

def compute_lighting(P, N, V, s):
	i = 0
	for light in scene.lights:
		if light.type == "ambient":
			i += light.intensity
		else:
			if light.type == "point":
				L = light.position - P
			else:
				L = light.direction

			# diffuse
			n_dot_l = np.dot(N, L)
			if n_dot_l > 0:
				i += light.intensity*n_dot_l/(np.linalg.norm(N)*np.linalg.norm(L))

			# specular
			if s != -1:
				R = 2*N*np.dot(N, L) - L
				r_dot_v = np.dot(R, V)
				if r_dot_v > 0:
					i += light.intensity*(r_dot_v/(np.linalg.norm(R)*np.linalg.norm(V)))**s

	return i

if __name__ == "__main__":
	O = np.array([0, 0, 0])

	canvas = np.zeros((C_h, C_w, 3)).astype(int)

	scene.spheres.append(Sphere(center=np.array([0, -1, 3]), radius=1, color=np.array([255, 0, 0]), specular=500))
	scene.spheres.append(Sphere(center=np.array([2, 0, 4]), radius=1, color=np.array([0, 0, 255]), specular=500))
	scene.spheres.append(Sphere(center=np.array([-2, 0, 4]), radius=1, color=np.array([0, 255, 0]), specular=10))
	scene.spheres.append(Sphere(center=np.array([0, -5001, 0]), radius=5000, color=np.array([255, 255, 0]), specular=1000))

	scene.lights.append(Light(type="ambient", intensity=0.2))
	scene.lights.append(Light(type="point", intensity=0.6, position=np.array([2, 1, 0])))
	scene.lights.append(Light(type="directional", intensity=0.2, direction=np.array([1, 4, 4])))

	for x in range(int(-C_w/2), int(C_w/2)):
		for y in range(int(-C_h/2), int(C_h/2)):
			D = canvas_to_viewport(x, y)
			color = trace_ray(O, D, 1, np.inf)

			canvas[y+int(C_h/2), x+int(C_w/2)] = color


	plt.imshow(np.flipud(canvas))
	plt.show()