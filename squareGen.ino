#include <TimerOne.h>

const int outputPin = 9;  // Pino de saída (porta 9)

float frequency;  // Variável para armazenar a frequência
float duration;   // Variável para armazenar a duração

void setup() {
  pinMode(outputPin, OUTPUT);  // Configura o pino como saída
  Serial.begin(9600);  // Inicia a comunicação serial
  Serial.println("Digite a frequencia em Hz e pressione Enter:");
}

void loop() {
  // Verifica se há entrada disponível no monitor serial
  if (Serial.available() > 0) {
    // Lê a frequência inserida pelo usuário
    frequency = readFrequency();
    
    // Verifica se a frequência é válida
    if (frequency > 0) {
      // Solicita o tempo de duração
      Serial.println("Digite o tempo de duracao em segundos e pressione Enter:");
      
      // Lê a duração inserida pelo usuário
      duration = readDuration();
      
      // Verifica se a duração é válida
      if (duration > 0) {
        // Gera a onda quadrada com base na frequência e duração fornecidas
        generateSquareWave(frequency, duration);
        
        // Solicita nova entrada de frequência e tempo
        Serial.println("Digite a frequencia em Hz e pressione Enter:");
      } else {
        Serial.println("Duracao invalida. Tente novamente.");
      }
    } else {
      Serial.println("Frequencia invalida. Tente novamente.");
    }
  }
}

// Função para ler a frequência do monitor serial
float readFrequency() {
  Serial.flush(); // Limpa o buffer de entrada
  while (Serial.available() == 0) {
    // Aguarda até que haja entrada disponível
  }
  float freq = Serial.parseFloat();
  // Garante que a leitura esteja completa
  while (Serial.available() > 0) {
    Serial.read(); // Limpa o buffer de entrada
  }
  return freq;
}

// Função para ler a duração do monitor serial
float readDuration() {
  Serial.flush(); // Limpa o buffer de entrada
  while (Serial.available() == 0) {
    // Aguarda até que haja entrada disponível
  }
  float dur = Serial.parseFloat();
  // Garante que a leitura esteja completa
  while (Serial.available() > 0) {
    Serial.read(); // Limpa o buffer de entrada
  }
  return dur;
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
