
# AppIA - Previsão de Preço de Imóveis (Exemplo)

Esta pasta contém um app Streamlit de exemplo para previsão de preço de imóveis (Boston-like).
Arquivos incluídos:
- app.py            -> aplicativo Streamlit
- data.csv          -> dataset de exemplo
- requirements.txt  -> dependências
- .vscode/launch.json -> configuração de depuração para VS Code

## Como usar (VS Code terminal)
1. Abra esta pasta no VS Code.
2. Crie e ative um virtualenv:
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
4. Rode a app:
   ```bash
   streamlit run app.py
   ```
5. Substitua `data.csv` pelos seus dados se desejar (garanta as colunas CRIM, INDUS, CHAS, NOX, RM, AGE, PTRATIO, MEDV).

## Integração com sua IA
- Se você já tem um modelo salvo (`.pkl`, `.joblib`, `.pt` etc), coloque-o na pasta e ajuste `load_model_or_train` em `app.py` para carregá-lo.
- Se precisar, peça aqui e eu adapto `app.py` para carregar seu arquivo específico.
