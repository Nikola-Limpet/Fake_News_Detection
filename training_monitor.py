# training_monitor.py
# Real-time training visualization and monitoring

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Training Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Fake News Detection - Training Monitor")
st.markdown("Real-time visualization of model training progress")

# Initialize session state
if 'training_active' not in st.session_state:
    st.session_state.training_active = False

if 'training_history' not in st.session_state:
    st.session_state.training_history = {
        'naive_bayes': {'status': 'Not started', 'metrics': {}},
        'random_forest': {'status': 'Not started', 'metrics': {}},
        'lstm': {'status': 'Not started', 'metrics': {}, 'history': []},
        'bert': {'status': 'Not started', 'metrics': {}, 'history': []}
    }

# Sidebar controls
with st.sidebar:
    st.markdown("## 🎛️ Training Controls")
    
    model_to_train = st.selectbox(
        "Select Model to Train",
        ["All Models", "Naive Bayes", "Random Forest", "LSTM", "BERT"]
    )
    
    st.markdown("### Training Parameters")
    
    epochs = st.slider("Epochs (for deep learning)", 1, 50, 10)
    batch_size = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
    learning_rate = st.select_slider(
        "Learning Rate",
        [0.0001, 0.001, 0.01, 0.1],
        value=0.001,
        format_func=lambda x: f"{x:.4f}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Training", type="primary", disabled=st.session_state.training_active):
            st.session_state.training_active = True
    
    with col2:
        if st.button("⏹️ Stop Training", type="secondary"):
            st.session_state.training_active = False
    
    st.markdown("---")
    
    # Training status
    st.markdown("### 📊 Training Status")
    for model, data in st.session_state.training_history.items():
        status = data['status']
        if status == 'Complete':
            st.success(f"✅ {model.replace('_', ' ').title()}")
        elif status == 'Training':
            st.warning(f"⏳ {model.replace('_', ' ').title()}")
        else:
            st.info(f"⭕ {model.replace('_', ' ').title()}")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Metrics", "🎯 Model Comparison", "📊 Detailed Analysis", "💾 Export Results"])

with tab1:
    st.markdown("### Real-time Training Metrics")
    
    # Create placeholders for live updates
    metric_container = st.container()
    chart_container = st.container()
    
    if st.session_state.training_active:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training process
        models_to_process = []
        if model_to_train == "All Models":
            models_to_process = ["naive_bayes", "random_forest", "lstm", "bert"]
        else:
            models_to_process = [model_to_train.lower().replace(" ", "_")]
        
        total_steps = len(models_to_process) * epochs if "lstm" in models_to_process or "bert" in models_to_process else len(models_to_process)
        current_step = 0
        
        for model in models_to_process:
            st.session_state.training_history[model]['status'] = 'Training'
            
            if model in ['naive_bayes', 'random_forest']:
                # Simulate quick training for classical ML
                status_text.text(f"Training {model.replace('_', ' ').title()}...")
                time.sleep(2)
                
                # Simulate metrics
                st.session_state.training_history[model]['metrics'] = {
                    'accuracy': np.random.uniform(0.85, 0.92),
                    'precision': np.random.uniform(0.84, 0.91),
                    'recall': np.random.uniform(0.83, 0.90),
                    'f1_score': np.random.uniform(0.84, 0.90),
                    'training_time': np.random.uniform(10, 60)
                }
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
            else:
                # Simulate epoch-based training for deep learning
                history = []
                for epoch in range(epochs):
                    status_text.text(f"Training {model.upper()} - Epoch {epoch+1}/{epochs}")
                    
                    # Simulate training metrics
                    train_loss = 0.7 * np.exp(-0.1 * epoch) + np.random.normal(0, 0.02)
                    val_loss = 0.75 * np.exp(-0.08 * epoch) + np.random.normal(0, 0.03)
                    train_acc = 0.6 + 0.35 * (1 - np.exp(-0.2 * epoch)) + np.random.normal(0, 0.01)
                    val_acc = 0.58 + 0.33 * (1 - np.exp(-0.18 * epoch)) + np.random.normal(0, 0.02)
                    
                    history.append({
                        'epoch': epoch + 1,
                        'train_loss': max(0, train_loss),
                        'val_loss': max(0, val_loss),
                        'train_acc': min(1, max(0, train_acc)),
                        'val_acc': min(1, max(0, val_acc))
                    })
                    
                    st.session_state.training_history[model]['history'] = history
                    
                    # Update live charts
                    with chart_container:
                        if len(history) > 1:
                            df_history = pd.DataFrame(history)
                            
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=(f'{model.upper()} - Loss', f'{model.upper()} - Accuracy')
                            )
                            
                            # Loss plot
                            fig.add_trace(
                                go.Scatter(x=df_history['epoch'], y=df_history['train_loss'],
                                         mode='lines', name='Train Loss'),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Scatter(x=df_history['epoch'], y=df_history['val_loss'],
                                         mode='lines', name='Val Loss'),
                                row=1, col=1
                            )
                            
                            # Accuracy plot
                            fig.add_trace(
                                go.Scatter(x=df_history['epoch'], y=df_history['train_acc'],
                                         mode='lines', name='Train Acc'),
                                row=1, col=2
                            )
                            fig.add_trace(
                                go.Scatter(x=df_history['epoch'], y=df_history['val_acc'],
                                         mode='lines', name='Val Acc'),
                                row=1, col=2
                            )
                            
                            fig.update_layout(height=400, showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Update metrics
                    with metric_container:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Train Loss", f"{train_loss:.4f}", f"{train_loss - history[-2]['train_loss']:.4f}" if len(history) > 1 else None)
                        col2.metric("Val Loss", f"{val_loss:.4f}", f"{val_loss - history[-2]['val_loss']:.4f}" if len(history) > 1 else None)
                        col3.metric("Train Acc", f"{train_acc:.2%}", f"{(train_acc - history[-2]['train_acc'])*100:.1f}%" if len(history) > 1 else None)
                        col4.metric("Val Acc", f"{val_acc:.2%}", f"{(val_acc - history[-2]['val_acc'])*100:.1f}%" if len(history) > 1 else None)
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    time.sleep(1)  # Simulate training time
                
                # Final metrics
                st.session_state.training_history[model]['metrics'] = {
                    'accuracy': history[-1]['val_acc'],
                    'precision': history[-1]['val_acc'] - np.random.uniform(0.01, 0.03),
                    'recall': history[-1]['val_acc'] - np.random.uniform(0.01, 0.04),
                    'f1_score': history[-1]['val_acc'] - np.random.uniform(0.01, 0.035),
                    'training_time': epochs * np.random.uniform(60, 180)
                }
            
            st.session_state.training_history[model]['status'] = 'Complete'
        
        st.session_state.training_active = False
        status_text.text("Training complete!")
        st.balloons()

with tab2:
    st.markdown("### Model Performance Comparison")
    
    # Collect metrics for comparison
    comparison_data = []
    for model, data in st.session_state.training_history.items():
        if data['status'] == 'Complete' and 'metrics' in data:
            comparison_data.append({
                'Model': model.replace('_', ' ').title(),
                **data['metrics']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Performance metrics bar chart
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        for metric in metrics_to_plot:
            if metric in df_comparison.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=df_comparison['Model'],
                    y=df_comparison[metric],
                    text=[f'{v:.2%}' for v in df_comparison[metric]],
                    textposition='outside'
                ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1.1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_time = go.Figure(data=[
                go.Bar(
                    x=df_comparison['Model'],
                    y=df_comparison['training_time'],
                    text=[f'{t:.0f}s' for t in df_comparison['training_time']],
                    textposition='outside',
                    marker_color=['red', 'orange', 'yellow', 'green']
                )
            ])
            fig_time.update_layout(
                title='Training Time Comparison',
                yaxis_title='Time (seconds)',
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Radar chart for multi-metric comparison
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            fig_radar = go.Figure()
            
            for _, row in df_comparison.iterrows():
                values = [row['accuracy'], row['precision'], row['recall'], row['f1_score']]
                values.append(values[0])  # Complete the circle
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Multi-Metric Comparison",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("### Detailed Metrics Table")
        
        # Format the dataframe for display
        df_display = df_comparison.copy()
        for col in ['accuracy', 'precision', 'recall', 'f1_score']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}')
        df_display['training_time'] = df_display['training_time'].apply(lambda x: f'{x:.1f}s')
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
    else:
        st.info("No completed training sessions yet. Start training to see comparisons.")

with tab3:
    st.markdown("### Detailed Training Analysis")
    
    # Select model for detailed analysis
    completed_models = [model for model, data in st.session_state.training_history.items() 
                       if data['status'] == 'Complete']
    
    if completed_models:
        selected_model = st.selectbox("Select Model for Analysis", completed_models)
        
        model_data = st.session_state.training_history[selected_model]
        
        if 'history' in model_data and model_data['history']:
            # Learning curves
            st.markdown(f"#### {selected_model.replace('_', ' ').title()} Learning Curves")
            
            df_history = pd.DataFrame(model_data['history'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Validation Loss', 
                              'Training Accuracy', 'Validation Accuracy'),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Training Loss
            fig.add_trace(
                go.Scatter(x=df_history['epoch'], y=df_history['train_loss'],
                         mode='lines+markers', name='Train Loss',
                         line=dict(color='blue')),
                row=1, col=1
            )
            
            # Validation Loss
            fig.add_trace(
                go.Scatter(x=df_history['epoch'], y=df_history['val_loss'],
                         mode='lines+markers', name='Val Loss',
                         line=dict(color='orange')),
                row=1, col=2
            )
            
            # Training Accuracy
            fig.add_trace(
                go.Scatter(x=df_history['epoch'], y=df_history['train_acc'],
                         mode='lines+markers', name='Train Acc',
                         line=dict(color='green')),
                row=2, col=1
            )
            
            # Validation Accuracy
            fig.add_trace(
                go.Scatter(x=df_history['epoch'], y=df_history['val_acc'],
                         mode='lines+markers', name='Val Acc',
                         line=dict(color='red')),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Convergence analysis
            st.markdown("#### Convergence Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_epoch = df_history['val_acc'].idxmax() + 1
                st.metric("Best Epoch", best_epoch)
            
            with col2:
                best_val_acc = df_history['val_acc'].max()
                st.metric("Best Validation Accuracy", f"{best_val_acc:.4f}")
            
            with col3:
                final_val_acc = df_history['val_acc'].iloc[-1]
                overfit_indicator = df_history['train_acc'].iloc[-1] - final_val_acc
                st.metric("Overfitting Indicator", f"{overfit_indicator:.4f}")
            
            # Early stopping analysis
            if len(df_history) > 5:
                patience = 3
                would_stop = None
                for i in range(patience, len(df_history)):
                    if all(df_history['val_acc'].iloc[i] <= df_history['val_acc'].iloc[i-j] 
                          for j in range(1, patience+1)):
                        would_stop = i
                        break
                
                if would_stop:
                    st.warning(f"Early stopping would have triggered at epoch {would_stop}")
                else:
                    st.success("No early stopping triggered - model continued improving")
        
        # Model-specific metrics
        st.markdown("#### Final Performance Metrics")
        
        metrics = model_data['metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    
    else:
        st.info("No completed training sessions for detailed analysis.")

with tab4:
    st.markdown("### Export Training Results")
    
    if any(data['status'] == 'Complete' for data in st.session_state.training_history.values()):
        
        # Prepare export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'training_parameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'models': {}
        }
        
        for model, data in st.session_state.training_history.items():
            if data['status'] == 'Complete':
                export_data['models'][model] = {
                    'metrics': data['metrics'],
                    'history': data.get('history', [])
                }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export metrics as CSV
            metrics_list = []
            for model, data in st.session_state.training_history.items():
                if data['status'] == 'Complete' and 'metrics' in data:
                    metrics_list.append({
                        'model': model,
                        **data['metrics']
                    })
            
            if metrics_list:
                df_export = pd.DataFrame(metrics_list)
                csv_str = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=csv_str,
                    file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate report
            if st.button("📄 Generate Report"):
                report = f"""
# Fake News Detection Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Parameters
- Epochs: {epochs}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}

## Model Performance Summary
"""
                for model, data in st.session_state.training_history.items():
                    if data['status'] == 'Complete' and 'metrics' in data:
                        metrics = data['metrics']
                        report += f"""
### {model.replace('_', ' ').title()}
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- Training Time: {metrics['training_time']:.1f} seconds
"""
                
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # Visualization of results
        st.markdown("### Training Summary Visualization")
        
        # Create summary visualization
        completed_models = [(model, data) for model, data in st.session_state.training_history.items() 
                          if data['status'] == 'Complete']
        
        if completed_models:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                              'Training Time', 'Precision vs Recall'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                      [{'type': 'bar'}, {'type': 'scatter'}]]
            )
            
            models = [m for m, _ in completed_models]
            model_names = [m.replace('_', ' ').title() for m in models]
            
            # Accuracy
            accuracies = [d['metrics']['accuracy'] for _, d in completed_models]
            fig.add_trace(
                go.Bar(x=model_names, y=accuracies, marker_color='blue'),
                row=1, col=1
            )
            
            # F1-Score
            f1_scores = [d['metrics']['f1_score'] for _, d in completed_models]
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, marker_color='green'),
                row=1, col=2
            )
            
            # Training Time
            train_times = [d['metrics']['training_time'] for _, d in completed_models]
            fig.add_trace(
                go.Bar(x=model_names, y=train_times, marker_color='orange'),
                row=2, col=1
            )
            
            # Precision vs Recall
            precisions = [d['metrics']['precision'] for _, d in completed_models]
            recalls = [d['metrics']['recall'] for _, d in completed_models]
            fig.add_trace(
                go.Scatter(x=precisions, y=recalls, mode='markers+text',
                         text=model_names, textposition="top center",
                         marker=dict(size=15, color=['red', 'blue', 'green', 'purple'][:len(models)])),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Precision", row=2, col=2)
            fig.update_yaxes(title_text="Recall", row=2, col=2)
            fig.update_yaxes(title_text="Seconds", row=2, col=1)
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No training results to export yet. Complete a training session first.")