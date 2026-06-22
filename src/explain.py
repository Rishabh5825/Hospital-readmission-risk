import shap
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, xgb_model, preprocessor):
        self.explainer = shap.TreeExplainer(xgb_model)
        self.preprocessor = preprocessor
        
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        self.feature_names = list(num_features) + list(cat_features)
        
    def explain_instance(self, df_row):
        """Generates top drivers and waterfall plot for a single patient."""
        transformed_data = self.preprocessor.transform(df_row)
        sv = self.explainer(transformed_data)
        
        feature_impacts = pd.DataFrame({'feature': self.feature_names, 'shap_value': sv[0].values})
        top_drivers_df = feature_impacts.reindex(feature_impacts['shap_value'].abs().sort_values(ascending=False).index).head(3)
        
        top_drivers = [
            {
                "feature": row['feature'],
                "impact": f"{'+' if row['shap_value'] > 0 else ''}{round(float(row['shap_value']), 3)}",
                "direction": "increases risk" if row['shap_value'] > 0 else "decreases risk"
            } for _, row in top_drivers_df.iterrows()
        ]
        
        # Generate plot
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(sv[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        return top_drivers, plot_base64
