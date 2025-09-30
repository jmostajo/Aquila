import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;

public class GarantiaRealDemo {

    public static void main(String[] args) {
        // Cambia la ruta aquí a la ruta absoluta de tu archivo en el escritorio
        String rutaArchivo = "/Users/juanjosemostajoleon/Desktop/Intereses2.xlsx"; 
        DateTimeFormatter formatoFecha = DateTimeFormatter.ofPattern("dd/MM/yy");

        try (FileInputStream file = new FileInputStream(rutaArchivo);
             Workbook workbook = new XSSFWorkbook(file)) {

            Sheet sheet = workbook.getSheetAt(0);

            // Asumimos que la primera fila es el encabezado
            Row encabezado = sheet.getRow(0);
            // Buscar o crear columna "Interés Total Devengado"
            int ultimaColumna = encabezado.getLastCellNum();
            int colInteresTotal = ultimaColumna; // Insertamos al final
            Cell celdaEncabezadoInteresTotal = encabezado.createCell(colInteresTotal);
            celdaEncabezadoInteresTotal.setCellValue("Interés Total Devengado (USD)");

            // Iterar filas con datos (desde fila 1)
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row fila = sheet.getRow(i);
                if (fila == null) continue;

                // Leer datos relevantes (basado en tu orden)
                Cell celdaFechaInicio = fila.getCell(1);
                Cell celdaFechaVencimiento = fila.getCell(2);
                Cell celdaInteresDiario = fila.getCell(3);

                if (celdaFechaInicio == null || celdaFechaVencimiento == null || celdaInteresDiario == null) {
                    System.out.println("Fila " + i + " incompleta. Se omite.");
                    continue;
                }

                String fechaInicioStr = celdaFechaInicio.getStringCellValue();
                String fechaVencimientoStr = celdaFechaVencimiento.getStringCellValue();

                LocalDate fechaInicio = LocalDate.parse(fechaInicioStr, formatoFecha);
                LocalDate fechaVencimiento = LocalDate.parse(fechaVencimientoStr, formatoFecha);
                LocalDate fechaAntesVencimiento = fechaVencimiento.minusDays(1);

                double interesDiario = 0;
                if (celdaInteresDiario.getCellType() == CellType.NUMERIC) {
                    interesDiario = celdaInteresDiario.getNumericCellValue();
                } else {
                    String val = celdaInteresDiario.getStringCellValue().replace("$", "").trim();
                    interesDiario = Double.parseDouble(val);
                }

                long diasDevengados = ChronoUnit.DAYS.between(fechaInicio, fechaAntesVencimiento) + 1;
                if (diasDevengados < 0) diasDevengados = 0;

                double interesTotal = diasDevengados * interesDiario;

                Cell celdaResultado = fila.createCell(colInteresTotal);
                celdaResultado.setCellValue(interesTotal);

                System.out.printf("Fila %d - Días: %d, Interés Total: %.2f USD%n", i, diasDevengados, interesTotal);
            }

            // Guardar archivo actualizado
            try (FileOutputStream outputStream = new FileOutputStream("/Users/juanjosemostajoleon/Desktop/intereses_actualizado.xlsx")) {
                workbook.write(outputStream);
            }

            System.out.println("\nArchivo procesado y guardado como 'intereses_actualizado.xlsx' en tu escritorio");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
