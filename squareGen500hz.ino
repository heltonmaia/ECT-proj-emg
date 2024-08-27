#include <TimerOne.h>

const int outputPin = 9;  // Pino de saída (porta 9)
const float frequency = 500;  // Frequência em Hz
const float duration = 50;    // Duração em segundos

void setup() {
  pinMode(outputPin, OUTPUT);  // Configura o pino como saída

  // Gera a onda quadrada com base na frequência e duração
  generateSquareWave(frequency, duration);
}

void loop() {
  // Nada a fazer no loop principal
}

// Função para gerar a onda quadrada com a frequência e duração especificadas
void generateSquareWave(float freq, float dur) {
  // Calcula o período em microssegundos (us) com base na frequência
  unsigned long period = 1000000.0 / freq;

  // Inicializa o Timer1 com o período calculado
  Timer1.initialize(period / 2);  // O Timer1 é configurado para metade do período (HIGH e LOW)
  Timer1.attachInterrupt(togglePin);  // Associa a função togglePin ao Timer1

  // Espera pelo tempo de duração especificado
  delay(dur * 1000);

  // Para o Timer1 após a duração especificada
  Timer1.detachInterrupt();
  digitalWrite(outputPin, LOW);  // Define o pino como LOW após parar o timer
}

// Função que alterna o estado do pino
void togglePin() {
  digitalWrite(outputPin, !digitalRead(outputPin));
}
