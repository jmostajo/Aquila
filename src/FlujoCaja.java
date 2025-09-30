import java.util.Arrays;

public class FlujoCaja {  // Cambié "Main" por "FlujoCaja"
    public static double calcularVarianza(double[] flujoCajaNeto) {
        double suma = 0.0;
        double media = 0.0;
        for (double d : flujoCajaNeto) {
            suma += d;
        }
        media = suma / flujoCajaNeto.length;

        double varianza = 0.0;
        for (double d : flujoCajaNeto) {
            varianza += Math.pow(d - media, 2);
        }
        return varianza / flujoCajaNeto.length;
    }

    public static double[] suavizadoExponencialDoble(double[] flujoCajaNeto, double alpha, double beta) {
        double[] suavizado = new double[flujoCajaNeto.length];
        double[] suavizadoDoble = new double[flujoCajaNeto.length];
        
        suavizado[0] = flujoCajaNeto[0];
        suavizadoDoble[0] = flujoCajaNeto[0];

        // Primer suavizado exponencial
        for (int i = 1; i < flujoCajaNeto.length; i++) {
            suavizado[i] = alpha * flujoCajaNeto[i] + (1 - alpha) * suavizado[i - 1];
        }

        // Segundo suavizado exponencial (suavizado doble)
        for (int i = 1; i < flujoCajaNeto.length; i++) {
            suavizadoDoble[i] = beta * suavizado[i] + (1 - beta) * suavizadoDoble[i - 1];
        }

        return suavizadoDoble;
    }

    public static void main(String[] args) {
        double[] flujoCajaNeto = {
            -456354.35, -279758.75, 73009.70, 264582.00, 0.00, 0.00,
            -859927.96, 157632.34, -127513.24, 261715.06, 3300.73, 1996641.34
        };

        // Calcular la varianza
        double varianza = calcularVarianza(flujoCajaNeto);
        System.out.println("La varianza del flujo de caja neto es: " + varianza);

        // Realizar el suavizado exponencial doble
        double alpha = 0.2; // Ajusta este valor según sea necesario
        double beta = 0.3;  // Ajusta este valor según sea necesario
        double[] flujoSuavizado = suavizadoExponencialDoble(flujoCajaNeto, alpha, beta);

        // Mostrar los resultados del suavizado exponencial doble
        System.out.println("Flujo de caja neto suavizado (exponencial doble): ");
        System.out.println(Arrays.toString(flujoSuavizado));
    }
}