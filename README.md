# Detector de deglutições

## Introdução
A disfagia é a dificuldade em engolir alimentos ou bebidas. Pode afetar a saúde e o bem-estar, por isso é importante detectar seus sinais precocemente. Idosos e pessoas com doenças como doença de Parkinson, doença de Alzheimer, demência, câncer de cabeça e pescoço ou histórico de derrame podem apresentar dificuldades para engolir.

![DisfagiaImagem](https://github.com/user-attachments/assets/dff85ad9-eb38-4abd-8429-e88bfb45f67c)
Figura 1 - Sinais e sintomas da disfagia.

A videofluoroscopia da deglutição (Figura 2) e a videoendoscopia da deglutição (Figura 3) são os exames mais comumente recomendados para avaliação da disfagia orofaríngea em lactentes e crianças. A investigação de pacientes com disfagia esofágica deve ser baseada na história clínica. Caso a história seja sugestiva de um distúrbio mecânico, a endoscopia digestiva alta ou o esofagograma de bário devem ser solicitados. Por outro lado, quando a história é sugestiva de um distúrbio de motilidade, a manometria é o teste de eleição para o diagnóstico.

![Videofluoroscopia](https://github.com/user-attachments/assets/e7ac8fdf-84af-4cbb-ad02-6a30ea2eca3e)

Figura 2 - Videofluoroscopia da deglutição

![Videoendoscopia](https://github.com/user-attachments/assets/1c2608a2-6444-4981-8c69-7e24af883a9c)

Figura 3 - Videoendoscopia da deglutição

### Novo Procedimento Não Invasivo
Um crescente contingente de tecnologias está atualmente disponível para avaliar os vários aspectos da função e disfunção da deglutição. Essas tecnologias permitem a mensuração dos movimentos das estruturas nela envolvidas e da ação muscular através de aspectos espaciais e temporais. Esse contingente de tecnologia pode fornecer diferentes peças analíticas de todo o processo. A escolha dos métodos para uma determinada avaliação será particularizada para cada caso, ou objetivo, dependendo das questões clínicas envolvidas.

Um procedimento não invasivo para avaliar a função de deglutição é utilizado um equipamento de Sonar Doppler. Esse procedimento usa um detector ultrassônico portátil para medir os movimentos das estruturas envolvidas na deglutição e a ação muscular, fornecendo informações sobre aspectos espaciais e temporais do processo. O transdutor é posicionado na região lateral da traqueia, abaixo da cartilagem cricoide, no lado direito do pescoço do paciente, com um ângulo de 30 a 60 graus. O equipamento é conectado a um microcomputador para processamento dos dados.

## Base de Dados
A base de dados foi gerada a partir de 490 exames, foram obtidos sinais sonoros provenientes de dopplers ultra sônicos fetais de baixo custo, realizando o procedimento descrito na introdução. 

## Objetivos
- Identificar na base de dados arquivos válidos e inválidos para treinamento e testes.
- Identificar os períodos de deglutição nos arquivos válidos
- Listar melhores metodologias de aprendizagem clássica por validação cruzada e busca de hiperparâmetros por métricas de acurácia balanceada para identificar arquivos válidos de inválidos.
- Utilizar modelos pré-treinados de aprendizagem profunda para identificação de segmentos de áudio com deglutições.
