import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.ChronoUnit;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;

public class GarantiaRealDemo {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== CÁLCULO DE INTERÉS DEVENGADO Y MORATORIO ===\n");

        // --- Sección fechas principales ---
        System.out.print("Ingrese la fecha de inicio (dd/MM/yyyy, dd-MM-yyyy o yyyy-MM-dd): ");
        LocalDate fechaInicio = parseFecha(scanner.nextLine().trim());

        System.out.print("Ingrese la fecha de vencimiento: ");
        LocalDate fechaVencimiento = parseFecha(scanner.nextLine().trim());

        System.out.print("Ingrese la fecha de pago: ");
        LocalDate fechaPago = parseFecha(scanner.nextLine().trim());

        if (fechaInicio == null || fechaVencimiento == null || fechaPago == null) {
            System.out.println("Error: Una o más fechas son inválidas.");
            System.exit(1);
        }

        if (fechaVencimiento.isBefore(fechaInicio)) {
            System.out.println("Error: La fecha de vencimiento no puede ser anterior a la fecha de inicio.");
            System.exit(1);
        }

        // --- Ingreso de datos financieros ---
        System.out.print("Ingrese el plazo operativo (en días): ");
        int plazoOperativo = leerEntero(scanner);

        System.out.print("Ingrese el interés al vencimiento I (USD): ");
        double interesVencimiento = leerDouble(scanner);

        double interesDevengadoDiario = interesVencimiento / plazoOperativo;
        long diasDevengados = ChronoUnit.DAYS.between(fechaInicio, fechaVencimiento);
        long diasMorosos = Math.max(0, ChronoUnit.DAYS.between(fechaVencimiento, fechaPago));

        System.out.print("Ingrese el monto total de la mora en USD: ");
        double interesMoratorioTotal = leerDouble(scanner);
        double interesMoratorioDiario = (diasMorosos > 0) ? interesMoratorioTotal / diasMorosos : 0.0;

        double interesDevengadoAcumulado = interesDevengadoDiario * diasDevengados;

        double interesMoratorioAcumulado = 0.0;
        LocalDate fechaActual = fechaVencimiento.plusDays(1);
        while (!fechaActual.isAfter(fechaPago)) {
            if (diasMorosos > 0) {
                interesMoratorioAcumulado += interesMoratorioDiario;
            }
            fechaActual = fechaActual.plusDays(1);
        }

        double interesTotalAcumulado = interesDevengadoAcumulado + interesMoratorioAcumulado;

        // --- Resultados generales en formato tabla ---
        System.out.println("\n====================== RESULTADOS GENERALES ======================");
        System.out.printf("%-45s %s%n", "Fecha de inicio:", fechaInicio);
        System.out.printf("%-45s %s%n", "Fecha de vencimiento:", fechaVencimiento);
        System.out.printf("%-45s %s%n", "Fecha de pago:", fechaPago);
        System.out.printf("%-45s %d días%n", "Plazo operativo:", plazoOperativo);
        System.out.printf("%-45s %.10f%n", "Interés devengado diario:", interesDevengadoDiario);
        System.out.printf("%-45s %d%n", "Días devengados:", diasDevengados);
        System.out.printf("%-45s $%.10f%n", "Interés devengado total acumulado:", interesDevengadoAcumulado);
        System.out.printf("%-45s %d%n", "Días morosos:", diasMorosos);
        System.out.printf("%-45s $%.2f%n", "Interés moratorio total ingresado:", interesMoratorioTotal);
        System.out.printf("%-45s %.10f%n", "Interés moratorio diario:", interesMoratorioDiario);
        System.out.printf("%-45s $%.10f%n", "Interés moratorio total acumulado:", interesMoratorioAcumulado);
        System.out.printf("%-45s $%.10f%n", "Interés total acumulado (devengado + moratorio):", interesTotalAcumulado);
        System.out.println("==================================================================");

        // --- CÁLCULO DE OVERNIGHT ---
        System.out.println("\n=== CÁLCULO DE INTERÉS ASOCIADO A OVERNIGHT ===");

        System.out.print("¿Desea calcular un préstamo overnight? (sí/no): ");
        String respuesta = scanner.nextLine().trim().toLowerCase();

        double montoOvernight = 0.0;
        double tasaInteresAnual = 0.0;
        LocalDate fechaInicioOvernight = null;
        LocalDate fechaVencimientoOvernight = null;
        long duracionOvernight = 0;
        double interesOvernight = 0.0;
        double interesDiarioOvernight = 0.0;

        if (respuesta.equals("sí") || respuesta.equals("si")) {

            System.out.print("Ingrese el monto del desembolso del overnight (USD): ");
            montoOvernight = leerDouble(scanner);

            System.out.print("Ingrese la tasa de interés anual (%) del overnight: ");
            tasaInteresAnual = leerDouble(scanner);
            double tasaInteresOvernight = tasaInteresAnual / 100.0;

            System.out.print("Ingrese la fecha de inicio del overnight: ");
            fechaInicioOvernight = parseFecha(scanner.nextLine().trim());

            System.out.print("Ingrese la fecha de vencimiento del overnight: ");
            fechaVencimientoOvernight = parseFecha(scanner.nextLine().trim());

            if (fechaInicioOvernight == null || fechaVencimientoOvernight == null) {
                System.out.println("Error: Fechas inválidas para el overnight.");
                System.exit(1);
            }

            duracionOvernight = ChronoUnit.DAYS.between(fechaInicioOvernight, fechaVencimientoOvernight);
            if (duracionOvernight <= 0) {
                System.out.println("Error: La duración del overnight debe ser mayor a 0.");
                System.exit(1);
            }

            interesOvernight = (Math.pow(1 + tasaInteresOvernight, duracionOvernight / 360.0) - 1) * montoOvernight;
            interesDiarioOvernight = interesOvernight / duracionOvernight;

            // Resultados overnight en formato tabla
            System.out.println("\n====================== RESULTADOS OVERNIGHT ======================");
            System.out.printf("%-45s $%.2f%n", "Monto del desembolso:", montoOvernight);
            System.out.printf("%-45s %.4f%%%n", "Tasa de interés anual:", tasaInteresAnual);
            System.out.printf("%-45s %s%n", "Fecha de inicio del overnight:", fechaInicioOvernight);
            System.out.printf("%-45s %s%n", "Fecha de vencimiento del overnight:", fechaVencimientoOvernight);
            System.out.printf("%-45s %d días%n", "Duración del overnight:", duracionOvernight);
            System.out.printf("%-45s $%.10f%n", "Interés total generado:", interesOvernight);
            System.out.printf("%-45s $%.10f%n", "Interés diario asociado:", interesDiarioOvernight);
            System.out.println("==================================================================");
        }

        // --- Generar archivo CSV con reporte ---
        String nombreArchivo = "garantiarealdemo.csv";
        try (FileWriter writer = new FileWriter(nombreArchivo)) {
            writer.append("Tipo de cálculo,Valor\n");
            writer.append("Fecha de inicio," + fechaInicio + "\n");
            writer.append("Fecha de vencimiento," + fechaVencimiento + "\n");
            writer.append("Fecha de pago," + fechaPago + "\n");
            writer.append("Plazo operativo," + plazoOperativo + "\n");
            writer.append("Interés devengado diario," + interesDevengadoDiario + "\n");
            writer.append("Días devengados," + diasDevengados + "\n");
            writer.append("Interés devengado total acumulado," + interesDevengadoAcumulado + "\n");
            writer.append("Días morosos," + diasMorosos + "\n");
            writer.append("Interés moratorio total ingresado," + interesMoratorioTotal + "\n");
            writer.append("Interés moratorio diario," + interesMoratorioDiario + "\n");
            writer.append("Interés moratorio total acumulado," + interesMoratorioAcumulado + "\n");
            writer.append("Interés total acumulado," + interesTotalAcumulado + "\n");

            if (respuesta.equals("sí") || respuesta.equals("si")) {
                writer.append("Monto del desembolso overnight," + montoOvernight + "\n");
                writer.append("Tasa de interés anual overnight," + tasaInteresAnual + "%\n");
                writer.append("Fecha inicio overnight," + fechaInicioOvernight + "\n");
                writer.append("Fecha vencimiento overnight," + fechaVencimientoOvernight + "\n");
                writer.append("Duración overnight," + duracionOvernight + "\n");
                writer.append("Interés total generado overnight," + interesOvernight + "\n");
                writer.append("Interés diario overnight," + interesDiarioOvernight + "\n");
            }

            System.out.println("\n📄 Report generated at: " + nombreArchivo);
        } catch (IOException e) {
            System.out.println("❌ Error al generar el archivo CSV: " + e.getMessage());
        }

        scanner.close();
    }

    // --- Funciones auxiliares ---
    private static LocalDate parseFecha(String fechaStr) {
        String[] formatos = {"dd/MM/yyyy", "dd-MM-yyyy", "yyyy-MM-dd"};
        for (String formato : formatos) {
            try {
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern(formato);
                return LocalDate.parse(fechaStr, formatter);
            } catch (DateTimeParseException ignored) {}
        }
        return null;
    }

    private static double leerDouble(Scanner scanner) {
        try {
            return Double.parseDouble(scanner.nextLine().trim().replace(",", "."));
        } catch (NumberFormatException e) {
            System.out.println("Error: Número inválido.");
            System.exit(1);
        }
        return 0.0;
    }

    private static int leerEntero(Scanner scanner) {
        try {
            return Integer.parseInt(scanner.nextLine().trim());
        } catch (NumberFormatException e) {
            System.out.println("Error: Debe ingresar un número entero válido.");
            System.exit(1);
        }
        return 0;
    }
}
