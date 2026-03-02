# ML Generic Project

## Descrição

Este projeto é um pipeline genérico de Machine Learning que demonstra a criação de um modelo preditivo e sua integração com uma aplicação web usando Flask. O objetivo principal é prever resultados de jogos (vitória do branco, vitória do preto ou empate) com base em classificações dos jogadores e nomes de aberturas de xadrez. O pipeline abrange desde a ingestão e transformação de dados até o treinamento do modelo e a disponibilização de previsões através de uma interface web.

## Funcionalidades

*   **Ingestão de Dados**: Processo para carregar dados brutos de um arquivo CSV.
*   **Transformação de Dados**: Pré-processamento dos dados, incluindo codificação de variáveis categóricas e escalonamento de características.
*   **Treinamento de Modelo**: Treinamento de um modelo de Machine Learning (XGBoost) para prever os resultados dos jogos.
*   **Previsão em Tempo Real**: Aplicação web Flask para receber entradas do usuário e fornecer previsões instantâneas.
*   **Estrutura Modular**: Código organizado em componentes, entidades e pipelines para facilitar a manutenção e escalabilidade.

## Estrutura do Projeto

```
ml-generic-project/
├── Dockerfile
├── README.md
├── application.py
├── artifacts/
│   ├── data_transformation/
│   │   └── preprocessor/
│   │       └── preprocessor.pkl
│   └── model_trainer/
│       └── model.pkl
├── data/
│   └── games.csv
├── main.py
├── noteboks/
│   └── EDA.ipynb
├── requirements.txt
├── setup.py
└── src/
    ├── Components/
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── Entity/
    │   ├── artifacts_config.py
    │   └── config_entity.py
    ├── Pipeline/
    │   ├── predict_pipeline.py
    │   └── train_pipeline.py
    ├── constants/
    ├── exception/
    ├── logging/
    └── utils/
```

## Tecnologias Utilizadas

*   **Python**: Linguagem de programação principal.
*   **Flask**: Framework web para a aplicação de previsão.
*   **Pandas**: Manipulação e análise de dados.
*   **NumPy**: Computação numérica.
*   **Scikit-learn**: Ferramentas para Machine Learning, incluindo pré-processamento.
*   **XGBoost**: Algoritmo de Machine Learning para treinamento do modelo.
*   **Dill**: Serialização de objetos Python.
*   **PyYAML**: Leitura de arquivos de configuração.
*   **Gunicorn**: Servidor WSGI para a aplicação Flask.

## Instalação

Para configurar e executar este projeto localmente, siga os passos abaixo:

1.  **Clone o repositório**:

    ```bash
    git clone https://github.com/beuren33/ml-generic-project.git
    cd ml-generic-project
    ```

2.  **Crie um ambiente virtual e ative-o**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

3.  **Instale as dependências**:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Treinamento do Modelo

Para treinar o modelo e gerar os artefatos de pré-processamento e modelo, execute o script `main.py`:

```bash
python main.py
```

Isso irá executar o pipeline de ingestão, transformação e treinamento de dados, salvando os artefatos necessários na pasta `artifacts/`.

### Executando a Aplicação Web

Após o treinamento do modelo, você pode iniciar a aplicação web Flask para fazer previsões:

```bash
python application.py
```

A aplicação estará disponível em `http://127.0.0.1:8000` (ou `http://0.0.0.0:8000` se executado em um contêiner).

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. (Assumindo licença MIT, caso contrário, ajustar.)

---

**Autor**: Manus AI
