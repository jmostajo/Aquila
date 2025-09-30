import java.util.LinkedHashMap;
import java.util.Map;

public class GarantiaRealDemo {

    // Parámetros del modelo
    private static double intercept = -3.0;
    private static double betaDPD = 0.01;
    private static double betaCobertura = -2.0;
    private static double betaExposicion = 0.05;
    private static double betaEscrow = -1.2; // NUEVO: efecto de flujos en cuentas escrow

    // Datos
    private static Map<String, Double> dpdClientes = new LinkedHashMap<>();
    private static Map<String, Double> deudaCapitalClientes = new LinkedHashMap<>();
    private static Map<String, Double> porcentajeExposicion = new LinkedHashMap<>();
    private static Map<String, Double> porcentajeFlujosEscrow = new LinkedHashMap<>(); // NUEVO
    private static Map<String, Integer> etiquetas = new LinkedHashMap<>();

    public static void main(String[] args) {
        inicializarDatos();
        entrenarModelo(10000, 0.000001);

        for (String cliente : dpdClientes.keySet()) {
            double dpd = dpdClientes.get(cliente);
            double deuda = deudaCapitalClientes.get(cliente);
            double porcentajeExp = porcentajeExposicion.get(cliente);
            double porcentajeEscrow = porcentajeFlujosEscrow.get(cliente); // NUEVO

            double ratioCoberturaNominal = calcularRatioCoberturaNominal(dpd);
            double garantiaNominal = deuda * ratioCoberturaNominal;
            double penalizacion = calcularPenalizacionPorDPD(dpd);
            double garantiaEjecutable = garantiaNominal * (1.0 - penalizacion);
            double coberturaReal = deuda != 0 ? garantiaEjecutable / deuda : 0;
            double riesgoResidual = deuda - garantiaEjecutable;

            double probDefault = calcularProbabilidadDefault(dpd, coberturaReal, porcentajeExp, porcentajeEscrow); // NUEVO
            int clasificacionBinaria = probDefault >= 0.5 ? 1 : 0;
            String garantiaReal = garantiaEjecutable > 0 ? "Sí" : "No";

            imprimirResultados(cliente, dpd, deuda, garantiaNominal, penalizacion, garantiaEjecutable, garantiaReal,
                    coberturaReal, riesgoResidual, porcentajeExp, porcentajeEscrow, probDefault, clasificacionBinaria);
        }
    }

    private static void inicializarDatos() {
        // DPD y deuda
        dpdClientes.put("MEDANOS", 0.0);
        dpdClientes.put("INKA'S", 29.0);
        dpdClientes.put("GANDULES", 68.166);
        dpdClientes.put("FRG", 0.0);
        dpdClientes.put("TRISKELION", 91.6);
        dpdClientes.put("COOPERATIVA ALONSO DE ALVARADO", 530.8);
        dpdClientes.put("LUBCOM", 1105.0);

        deudaCapitalClientes.put("MEDANOS", 2_000_000.42);
        deudaCapitalClientes.put("INKA'S", 3_150_000.00);
        deudaCapitalClientes.put("GANDULES", 1_680_000.00);
        deudaCapitalClientes.put("FRG", 500_000.00);
        deudaCapitalClientes.put("TRISKELION", 186_028.67);
        deudaCapitalClientes.put("COOPERATIVA ALONSO DE ALVARADO", 928_664.07);
        deudaCapitalClientes.put("LUBCOM", 18_498.38);

        porcentajeExposicion.put("MEDANOS", 10.70);
        porcentajeExposicion.put("INKA'S", 16.86);
        porcentajeExposicion.put("GANDULES", 8.99);
        porcentajeExposicion.put("FRG", 0.00);
        porcentajeExposicion.put("TRISKELION", 1.00);
        porcentajeExposicion.put("COOPERATIVA ALONSO DE ALVARADO", 4.97);
        porcentajeExposicion.put("LUBCOM", 0.10);

        porcentajeFlujosEscrow.put("MEDANOS", 60.0);
        porcentajeFlujosEscrow.put("INKA'S", 30.0);
        porcentajeFlujosEscrow.put("GANDULES", 0.0);
        porcentajeFlujosEscrow.put("FRG", 50.0);
        porcentajeFlujosEscrow.put("TRISKELION", 5.0);
        porcentajeFlujosEscrow.put("COOPERATIVA ALONSO DE ALVARADO", 0.0);
        porcentajeFlujosEscrow.put("LUBCOM", 0.0);

        etiquetas.put("MEDANOS", 0);
        etiquetas.put("INKA'S", 0);
        etiquetas.put("GANDULES", 0);
        etiquetas.put("FRG", 0);
        etiquetas.put("TRISKELION", 1);
        etiquetas.put("COOPERATIVA ALONSO DE ALVARADO", 1);
        etiquetas.put("LUBCOM", 1);
    }

    private static double calcularRatioCoberturaNominal(double dpd) {
        return dpd < 90 ? 2.0 - (dpd / 180.0) : 1.5;
    }

    public static double calcularPenalizacionPorDPD(double dpd) {
        if (dpd <= 30) return 0.10;
        else if (dpd <= 90) return 0.15;
        else if (dpd <= 365) return 0.20;
        else return 0.30;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double calcularProbabilidadDefault(double dpd, double coberturaReal, double porcentajeExposicion, double 
porcentajeEscrow) {
        // Primero calculamos la probabilidad sin efecto escrow
        double zSinEscrow = intercept + betaDPD * dpd + betaCobertura * coberturaReal + betaExposicion * porcentajeExposicion;
        double probSinEscrow = sigmoid(zSinEscrow);

        // Atenuación: cuanto más cerca de 1 sea probSinEscrow, menor peso tendrá el efecto escrow
        double atenuacionEscrow = 1.0 - Math.pow(probSinEscrow, 5);  // Exponente 5 para una caída rápida cerca de 1

        double z = zSinEscrow + betaEscrow * (porcentajeEscrow / 100.0) * atenuacionEscrow;

        return sigmoid(z);
    }

    public static void entrenarModelo(int maxIter, double learningRate) {
        for (int iter = 0; iter < maxIter; iter++) {
            double gradIntercept = 0, gradDPD = 0, gradCobertura = 0, gradExposicion = 0, gradEscrow = 0;
            int n = dpdClientes.size();

            for (String cliente : dpdClientes.keySet()) {
                double dpd = dpdClientes.get(cliente);
                double deuda = deudaCapitalClientes.get(cliente);
                double porcentajeExp = porcentajeExposicion.get(cliente);
                double porcentajeEscrow = porcentajeFlujosEscrow.get(cliente);
                double ratioCoberturaNominal = calcularRatioCoberturaNominal(dpd);
                double penalizacion = calcularPenalizacionPorDPD(dpd);
                double garantiaNominal = deuda * ratioCoberturaNominal;
                double garantiaEjecutable = garantiaNominal * (1.0 - penalizacion);
                double coberturaReal = deuda != 0 ? garantiaEjecutable / deuda : 0;

                int y = etiquetas.get(cliente);
                double pred = calcularProbabilidadDefault(dpd, coberturaReal, porcentajeExp, porcentajeEscrow);
                double error = pred - y;

                gradIntercept += error;
                gradDPD += error * dpd;
                gradCobertura += error * coberturaReal;
                gradExposicion += error * porcentajeExp;
                gradEscrow += error * (porcentajeEscrow / 100.0);
            }

            intercept -= learningRate * gradIntercept / n;
            betaDPD -= learningRate * gradDPD / n;
            betaCobertura -= learningRate * gradCobertura / n;
            betaExposicion -= learningRate * gradExposicion / n;
            betaEscrow -= learningRate * gradEscrow / n;

            if (iter % 1000 == 0) {
                double loss = calcularLogLoss();
                System.out.printf("Iteración %d, LogLoss: %.6f%n", iter, loss);
            }
        }
    }

    public static double calcularLogLoss() {
        double loss = 0;
        int n = dpdClientes.size();
        for (String cliente : dpdClientes.keySet()) {
            double dpd = dpdClientes.get(cliente);
            double deuda = deudaCapitalClientes.get(cliente);
            double porcentajeExp = porcentajeExposicion.get(cliente);
            double porcentajeEscrow = porcentajeFlujosEscrow.get(cliente);
            double ratioCoberturaNominal = calcularRatioCoberturaNominal(dpd);
            double penalizacion = calcularPenalizacionPorDPD(dpd);
            double garantiaEjecutable = deuda * ratioCoberturaNominal * (1.0 - penalizacion);
            double coberturaReal = deuda != 0 ? garantiaEjecutable / deuda : 0;

            int y = etiquetas.get(cliente);
            double pred = calcularProbabilidadDefault(dpd, coberturaReal, porcentajeExp, porcentajeEscrow);
            loss += -y * Math.log(pred + 1e-10) - (1 - y) * Math.log(1 - pred + 1e-10);
        }
        return loss / n;
    }

    private static void imprimirResultados(String cliente, double dpd, double deuda, double garantiaNominal, double penalizacion, double 
garantiaEjecutable,
                                           String garantiaReal, double coberturaReal, double riesgoResidual, double porcentajeExposicion, 
double porcentajeEscrow,
                                           double probDefault, int clasificacionBinaria) {
        System.out.println("Cliente: " + cliente);
        System.out.printf(" DPD: %.2f días%n", dpd);
        System.out.printf(" Deuda Capital Real: USD %.2f%n", deuda);
        System.out.printf(" Garantía Nominal: USD %.2f%n", garantiaNominal);
        System.out.printf(" Factor no ejecutable de la garantía: %.2f%%%n", penalizacion * 100);
        System.out.printf(" Garantía Ejecutable: USD %.2f%n", garantiaEjecutable);
        System.out.printf(" ¿Tiene Garantía Real?: %s%n", garantiaReal);
        System.out.printf(" Ratio de Cobertura Real: %.2f%n", coberturaReal);
        System.out.printf(" Riesgo Residual: USD %.2f%n", riesgoResidual);
        System.out.printf(" Porcentaje de Exposición: %.2f%%%n", porcentajeExposicion);
        System.out.printf(" %% Flujos a Cuentas Escrow: %.2f%%%n", porcentajeEscrow);
        System.out.printf(" Probabilidad de Default (modelo logístico): %.4f%n", probDefault);
        System.out.printf(" Clasificación Binaria (1 = riesgo alto): %d%n", clasificacionBinaria);
        System.out.println("--------------------------------------------------");
    }
}
