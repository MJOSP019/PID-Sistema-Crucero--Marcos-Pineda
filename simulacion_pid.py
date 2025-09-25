import pygame
import numpy as np
import math
import matplotlib.pyplot as plt

# Parámetros iniciales
V_MAX = 150  # Velocidad máxima (km/h)
WIDTH, HEIGHT = 800, 600  # Tamaño de la ventana
CAR_WIDTH, CAR_HEIGHT = 50, 30  # Tamaño del auto

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Limites de aceleración
MAX_ACCELERATION = 10  # Aceleración máxima en m/s²
MIN_ACCELERATION = -10  # Aceleración mínima en m/s²

# Función para iniciar Pygame
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulación de Control Crucero")
    return screen

# Clase para representar el auto
class Car:
    def __init__(self, x, y, speed=0):
        self.x = x
        self.y = y
        self.speed = speed  # Velocidad en m/s
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
    def update(self, target_speed, acceleration, dt, slope_angle):
        """Actualiza la velocidad y posición del auto según la aceleración y pendiente."""
        # Fuerza adicional debido a la pendiente (perturbación)
        g = 9.81  # Gravedad (m/s²)
        # Calculamos la fuerza de resistencia debido a la pendiente
        resistance_force = g * np.sin(np.radians(slope_angle))
        
        # Actualizamos la velocidad con la aceleración
        self.speed += (acceleration - resistance_force) * dt  # Ajuste por resistencia
        
        # Limitar la velocidad máxima
        self.speed = np.clip(self.speed, 0, target_speed)  # Limitar la velocidad máxima al objetivo

        # Actualizamos la posición
        self.x += self.speed * dt  # Actualizar la posición en X

        # Limitar el desplazamiento en X para que no se salga de la pantalla
        if self.x < 0:
            self.x = 0
        elif self.x > WIDTH:
            self.x = WIDTH

        self.rect.center = (self.x, self.y)

    def draw(self, screen):
        """Dibuja el auto en la pantalla."""
        screen.blit(self.image, self.rect)

# Controlador PID
def pid_controller(error, integral, derivative, Kp=0.36, Ki=0.18, Kd=0.02):
    """Controlador PID para el auto."""
    u = Kp * error + Ki * integral + Kd * derivative
    # Limitar la aceleración a un rango razonable
    return np.clip(u, MIN_ACCELERATION, MAX_ACCELERATION)

# Simulación
def simulate(car, pid_controller, dt=0.01, total_time=30):
    screen = init_pygame()
    clock = pygame.time.Clock()

    # Variables de control
    v_star = 90 / 3.6  # Velocidad objetivo en m/s (90 km/h)
    error = v_star - car.speed
    integral = 0
    prev_error = 0

    # Variable para la pendiente
    slope_angle = 0.0  # Pendiente inicial en grados

    # Listas para guardar la historia de la aceleración y la velocidad
    u_history = []
    v_history = []

    # Fuente de texto para mostrar aceleración
    font = pygame.font.Font(None, 36)

    # Bucle de la simulación
    running = True
    time = 0
    while running and time < total_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Manejo de la pendiente con teclas
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:  # Incrementar pendiente
            slope_angle += 0.1
        if keys[pygame.K_DOWN]:  # Decrecer pendiente
            slope_angle -= 0.1

        # Controlador PID
        error = v_star - car.speed
        integral += error * dt
        derivative = (error - prev_error) / dt
        u = pid_controller(error, integral, derivative)
        prev_error = error

        # Asegurarnos de que el error se mantenga bajo control
        if abs(error) < 0.1:
            u = 0.1  # Ajuste mínimo para evitar que el auto se quede parado

        # Actualizar la velocidad del auto con la perturbación de pendiente
        car.update(v_star, u, dt, slope_angle)
        
        # Guardar la historia de la aceleración y la velocidad
        u_history.append(u)
        v_history.append(car.speed * 3.6)  # Convertir a km/h

        # Actualizar la pantalla
        screen.fill(WHITE)
        car.draw(screen)

        # Mostrar información
        text = font.render(f"Velocidad: {car.speed*3.6:.2f} km/h", True, BLACK)
        screen.blit(text, (10, 10))

        # Mostrar pendiente actual
        slope_text = font.render(f"Pendiente: {slope_angle:.2f}°", True, BLACK)
        screen.blit(slope_text, (10, 50))

        # Mostrar la aceleración en tiempo real
        acceleration_text = font.render(f"Aceleración: {u:.2f} m/s²", True, BLACK)
        screen.blit(acceleration_text, (10, 90))

        # Actualizar la ventana
        pygame.display.flip()
        clock.tick(60)

        time += dt

    pygame.quit()

    # Graficar la historia de la señal de control y la velocidad
    plot_results(u_history, v_history)

# Graficar la historia de la señal de control y la velocidad
def plot_results(u_history, v_history):
    plt.figure(figsize=(10, 6))

    # Gráfico de la señal de control
    plt.subplot(2, 1, 1)
    plt.plot(u_history, label="Señal de control (u(t))")
    plt.title("Señal de control (Aceleración)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Aceleración [m/s²]")
    plt.grid(True)
    plt.legend()

    # Gráfico de la velocidad
    plt.subplot(2, 1, 2)
    plt.plot(v_history, label="Velocidad (v(t))")
    plt.title("Velocidad del auto")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [km/h]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Simular
if __name__ == "__main__":
    car = Car(100, HEIGHT // 2)  # Posición inicial en X, Y
    simulate(car, pid_controller)  # Simulación usando el PID
