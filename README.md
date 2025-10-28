
# Mushroom IA Project

Este projeto contém um app Streamlit que treina um classificador de cogumelos (comestível vs venenoso) a partir de um arquivo `mushroom.csv`.

## Como usar (VS Code terminal)

1. Coloque seu `mushroom.csv` na pasta do projeto (substitua o `mushroom.csv` de exemplo se desejar).
2. Crie e ative o ambiente virtual:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Rode o app:
   ```bash
   streamlit run app.py
   ```

## Observações
- O app detecta automaticamente colunas boolean-like e colunas categóricas (char).
- Treina um RandomForestClassifier dentro do app quando você clicar em 'Treinar modelo agora'.
- Após treinar, use os controles na sidebar para inserir características do cogumelo e obter uma previsão.
