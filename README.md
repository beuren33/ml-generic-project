### ML Generic Project

## Descrição

Este projeto é um pipeline genérico de Machine Learning que demonstra a criação de um modelo preditivo e sua integração com uma aplicação web usando Flask. O objetivo principal é prever resultados de jogos (vitória do branco, vitória do preto ou empate) com base em classificações dos jogadores e nomes de aberturas de xadrez. O pipeline abrange desde a ingestão e transformação de dados até o treinamento do modelo e a disponibilização de previsões através de uma interface web.

## Funcionalidades

*   **Ingestão de Dados**: Processo para carregar dados brutos de um arquivo CSV.
*   **Transformação de Dados**: Pré-processamento dos dados, incluindo codificação de variáveis categóricas e escalonamento de características.
*   **Treinamento de Modelo**: Com a implementação do RandomSearchCV foi selecionado o modelo com a melhor performace.
*   **Previsão em Tempo Real**: Aplicação web Flask para receber entradas do usuário e fornecer previsões instantâneas.
*   **Estrutura Modular**: Código organizado em componentes, entidades e pipelines para facilitar a manutenção e escalabilidade.


## Tecnologias Utilizadas

*   **Python**: Linguagem de programação principal.
*   **Flask**: Framework web para a aplicação de previsão.
*   **Pandas**: Manipulação e análise de dados.
*   **NumPy**: Computação numérica.
*   **Scikit-learn**: Ferramentas para Machine Learning, incluindo pré-processamento.
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

## Implantação (CI/CD)

Este projeto utiliza um pipeline de Continuous Integration/Continuous Deployment (CI/CD) na AWS para automatizar a construção, teste e implantação da aplicação. A infraestrutura de CI/CD é composta pelos seguintes serviços:

*   **Docker**: A aplicação é conteinerizada usando Docker, garantindo um ambiente consistente em todas as etapas do pipeline. O `Dockerfile` define como a imagem da aplicação é construída.

*   **AWS CodePipeline**: Orquestra todo o fluxo de CI/CD, desde a detecção de alterações no repositório GitHub até a implantação final no Elastic Beanstalk.

*   **AWS CodeBuild**: Utilizado pelo CodePipeline para construir a imagem Docker da aplicação, fazer login no Amazon ECR (Elastic Container Registry), enviar a imagem para o ECR e gerar o arquivo `Dockerrun.aws.json` necessário para a implantação no Elastic Beanstalk. O arquivo `buildspec.yml` especifica os comandos de build.

*   **Amazon Elastic Beanstalk**: Serviço de plataforma como serviço (PaaS) que facilita a implantação e o gerenciamento de aplicações web. Ele provisiona e gerencia a infraestrutura subjacente (servidores, balanceadores de carga, etc.) e implanta a imagem Docker da aplicação, conforme especificado no `Dockerrun.aws.json`.

### Fluxo de Implantação

1.  Um push para o repositório GitHub aciona o AWS CodePipeline.
2.  O CodePipeline inicia uma fase de build no AWS CodeBuild.
3.  O CodeBuild constrói a imagem Docker da aplicação, faz o push para o Amazon ECR e gera o `Dockerrun.aws.json`.
4.  O CodePipeline então implanta o artefato (`Dockerrun.aws.json`) no ambiente do Amazon Elastic Beanstalk.
5.  O Elastic Beanstalk puxa a imagem Docker do Amazon ECR e implanta a aplicação nos servidores, tornando-a acessível.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. (Assumindo licença MIT, caso contrário, ajustar.)

---

**Autor**: Matheus Beuren
