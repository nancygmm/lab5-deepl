import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_prep import load_sunspots, normalize_series, split_series
from train import train_rnn, make_windows
from model import RNNForecast

def detectar_gradientes_problema(model, dataloader, device="cpu"):
    model.train()
    lossf = nn.MSELoss()
    gradients = []
    
    for i, (xb, yb) in enumerate(dataloader):
        if i >= 5:  
            break
            
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        pred = model(xb)
        loss = lossf(pred, yb)
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradients.append(total_norm)
    
    grad_mean = np.mean(gradients)
    grad_std = np.std(gradients)
    
    vanishing = grad_mean < 1e-6
    exploding = grad_mean > 10.0 or grad_std > 5.0
    
    return {
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'vanishing': vanishing,
        'exploding': exploding,
        'gradients': gradients
    }

def evaluar_modelo(model, X_test, y_test, scaler, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred = model(X_test_tensor).cpu().numpy()
    
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_inv = scaler.inverse_transform(pred).flatten()
    
    mse = np.mean((y_test_inv - pred_inv) ** 2)
    mae = np.mean(np.abs(y_test_inv - pred_inv))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_test_inv - pred_inv) ** 2)
    ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_real': y_test_inv,
        'y_pred': pred_inv
    }

def experimento_ventana(seq_len, series_norm_train, series_norm_test, scaler, 
                       hidden_size=32, epochs=50, lr=1e-3, device="cpu"):

    print(f"\n{'='*50}")
    print(f"EXPERIMENTO: Ventana de {seq_len} pasos")
    print(f"{'='*50}")
    
    X_train, y_train = make_windows(series_norm_train, seq_len)
    X_test, y_test = make_windows(series_norm_test, seq_len)
    
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")
    
    start_time = time.time()
    model, best_loss = train_rnn(
        train_series=series_norm_train, 
        seq_len=seq_len, 
        hidden_size=hidden_size, 
        epochs=epochs, 
        lr=lr,
        device=device
    )
    training_time = time.time() - start_time
    
    from torch.utils.data import TensorDataset, DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                           batch_size=64, shuffle=True)
    
    grad_info = detectar_gradientes_problema(model, dataloader, device)
    
    metrics = evaluar_modelo(model, X_test, y_test, scaler, device)
    
    resultado = {
        'seq_len': seq_len,
        'training_time': training_time,
        'best_loss': best_loss,
        'mse': metrics['mse'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'grad_mean': grad_info['grad_mean'],
        'grad_std': grad_info['grad_std'],
        'vanishing': grad_info['vanishing'],
        'exploding': grad_info['exploding'],
        'y_real': metrics['y_real'],
        'y_pred': metrics['y_pred'],
        'gradients': grad_info['gradients']
    }
    
    print(f"Tiempo de entrenamiento: {training_time:.2f}s")
    print(f"Best training loss: {best_loss:.6f}")
    print(f"Test MSE: {metrics['mse']:.2f}")
    print(f"Test MAE: {metrics['mae']:.2f}")
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Gradiente promedio: {grad_info['grad_mean']:.6f}")
    print(f"Vanishing gradients: {'SÍ' if grad_info['vanishing'] else 'NO'}")
    print(f"Exploding gradients: {'SÍ' if grad_info['exploding'] else 'NO'}")
    
    return resultado, model

def crear_visualizaciones(resultados, out_dir):
    ventanas = [r['seq_len'] for r in resultados]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, resultado in enumerate(resultados):
        ax = axes[i]
        seq_len = resultado['seq_len']
        y_real = resultado['y_real']
        y_pred = resultado['y_pred']
        
        n_points = min(100, len(y_real))
        ax.plot(y_real[:n_points], label='Real', alpha=0.8)
        ax.plot(y_pred[:n_points], label='Predicción', alpha=0.8)
        ax.set_title(f'Ventana {seq_len} - R²={resultado["r2"]:.3f}')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Manchas solares')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "predicciones_comparativas.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].bar(ventanas, [r['mse'] for r in resultados], color='skyblue')
    axes[0,0].set_title('Error Cuadrático Medio (MSE)')
    axes[0,0].set_xlabel('Tamaño de ventana')
    axes[0,0].set_ylabel('MSE')
    
    axes[0,1].bar(ventanas, [r['r2'] for r in resultados], color='lightgreen')
    axes[0,1].set_title('Coeficiente de Determinación (R²)')
    axes[0,1].set_xlabel('Tamaño de ventana')
    axes[0,1].set_ylabel('R²')
    
    axes[1,0].bar(ventanas, [r['training_time'] for r in resultados], color='orange')
    axes[1,0].set_title('Tiempo de Entrenamiento')
    axes[1,0].set_xlabel('Tamaño de ventana')
    axes[1,0].set_ylabel('Tiempo (s)')
    
    axes[1,1].bar(ventanas, [r['grad_mean'] for r in resultados], color='coral')
    axes[1,1].set_title('Magnitud Promedio de Gradientes')
    axes[1,1].set_xlabel('Tamaño de ventana')
    axes[1,1].set_ylabel('Norma L2')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(out_dir / "metricas_comparativas.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, resultado in enumerate(resultados):
        ax = axes[i]
        gradients = resultado['gradients']
        ax.plot(gradients, marker='o', markersize=3)
        ax.set_title(f'Ventana {resultado["seq_len"]} - Gradientes por Batch')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Norma L2 del gradiente')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
        ax.axhline(y=10.0, color='orange', linestyle='--', alpha=0.7, label='Exploding threshold')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "analisis_gradientes.png", dpi=150, bbox_inches="tight")
    plt.close()

def crear_tabla_resultados(resultados, out_dir):

    datos_tabla = []
    for r in resultados:
        datos_tabla.append({
            'Ventana': r['seq_len'],
            'Tiempo_Entrenamiento(s)': round(r['training_time'], 2),
            'Training_Loss': round(r['best_loss'], 6),
            'Test_MSE': round(r['mse'], 2),
            'Test_MAE': round(r['mae'], 2),
            'Test_RMSE': round(r['rmse'], 2),
            'Test_R2': round(r['r2'], 4),
            'Grad_Mean': f"{r['grad_mean']:.2e}",
            'Grad_Std': f"{r['grad_std']:.2e}",
            'Vanishing_Grad': 'SÍ' if r['vanishing'] else 'NO',
            'Exploding_Grad': 'SÍ' if r['exploding'] else 'NO'
        })
    
    df = pd.DataFrame(datos_tabla)
    
    df.to_csv(out_dir / "resultados_experimentos.csv", index=False)
    
    print(f"\n{'='*100}")
    print("TABLA DE RESULTADOS COMPARATIVOS")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    
    return df

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    years, series = load_sunspots()
    series_norm, scaler = normalize_series(series)
    series_norm_train, series_norm_test = split_series(series_norm, train_ratio=0.8)
    
    print(f"Datos originales: {len(series)} puntos")
    print(f"Entrenamiento: {len(series_norm_train)} puntos")
    print(f"Prueba: {len(series_norm_test)} puntos")
    
    ventanas = [5, 10, 20, 100]
    
    resultados = []
    modelos = {}
    
    for seq_len in ventanas:
        try:
            resultado, modelo = experimento_ventana(
                seq_len=seq_len,
                series_norm_train=series_norm_train,
                series_norm_test=series_norm_test,
                scaler=scaler,
                hidden_size=32,
                epochs=50,
                lr=1e-3,
                device=device
            )
            resultados.append(resultado)
            modelos[seq_len] = modelo
            
        except Exception as e:
            print(f"Error en experimento con ventana {seq_len}: {e}")
            continue
    

    crear_visualizaciones(resultados, out_dir)

    df_resultados = crear_tabla_resultados(resultados, out_dir)
    
    print(f"\n{'='*60}")
    print("ANÁLISIS FINAL")
    print(f"{'='*60}")
    
    mejor_r2 = max(resultados, key=lambda x: x['r2'])
    menor_mse = min(resultados, key=lambda x: x['mse'])
    mas_rapido = min(resultados, key=lambda x: x['training_time'])
    
    print(f"Mejor R² (precisión): Ventana {mejor_r2['seq_len']} con R²={mejor_r2['r2']:.4f}")
    print(f"Menor MSE: Ventana {menor_mse['seq_len']} con MSE={menor_mse['mse']:.2f}")
    print(f"Entrenamiento más rápido: Ventana {mas_rapido['seq_len']} con {mas_rapido['training_time']:.2f}s")
    
    problemas_vanishing = [r for r in resultados if r['vanishing']]
    problemas_exploding = [r for r in resultados if r['exploding']]
    
    if problemas_vanishing:
        ventanas_problemas = [r['seq_len'] for r in problemas_vanishing]
        print(f"Problemas de vanishing gradients en ventanas: {ventanas_problemas}")
    
    if problemas_exploding:
        ventanas_problemas = [r['seq_len'] for r in problemas_exploding]
        print(f"Problemas de exploding gradients en ventanas: {ventanas_problemas}")
    
    print(f"\nArchivos generados en: {out_dir}")

if __name__ == "__main__":
    main()