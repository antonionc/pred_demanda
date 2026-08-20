# pred_demanda
Electricity Demand Prediction in Spain

Comparison of different AI & Machine Learning algorithms (SARIMAX, LSTM, and Tiny Time Mixers - TTM) for short-term electricity load forecasting in Spain.

## Data

We download hourly demand data from [www.esios.ree.es](https://www.esios.ree.es/es/generacion-y-consumo) (indicator 1293) and meteorological features from Open-Meteo across major Spanish cities.

The ESIOS API is public, but requires an API token:
* [API Token Request Information](https://www.esios.ree.es/es/pagina/api#)
* [API Documentation](https://api.esios.ree.es/)

### ESIOS API Key Configuration
The API token can be provided in any of the following ways:
1. **Google Colab Secrets**: Add a secret named `ESIOS_API_KEY`.
2. **Environment Variable**: `export ESIOS_API_KEY="your_token_here"`
3. **Local File**: A text file named `esios_api_key.txt` in the root folder.
4. **Google Drive**: Stored at `MyDrive/pred_demanda/esios_api_key.txt`.

## Running in Google Colab & Persistent Google Drive Storage

When executing in Google Colab, ephemeral disk storage is wiped when a runtime disconnects. To persist downloaded chunks, processed datasets, and trained models:
- Google Drive is automatically mounted at `/content/drive`.
- The workspace directories (`data/`, `cache/`, `saved_models/`) are automatically symlinked to `MyDrive/pred_demanda/` on Google Drive via `data_utils.setup_colab_drive()`.
- Re-running notebooks will automatically reuse cached downloads and previously computed files without re-downloading or re-processing.

## Project Notebooks

1. **`fase1_adquisicion_y_eda.ipynb`**: Data acquisition from ESIOS & Open-Meteo, feature engineering, and EDA.
2. **`fase2_1_sarimax.ipynb`**: SARIMAX statistical baseline model.
3. **`fase2_2_lstm.ipynb`**: Deep Learning 2-layer stacked LSTM network.
4. **`fase2_3_ttm.ipynb`**: IBM Granite Tiny Time Mixers (zero-shot & few-shot fine-tuning).
5. **`fase3_validacion_y_comparativa.ipynb`**: Model validation, benchmark comparisons, and metrics evaluation.

