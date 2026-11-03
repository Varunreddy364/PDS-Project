library(ggplot2)

# Figure 1: 10 Models R-Squared
df <- data.frame(
  Model = factor(c('Decision Tree', 'Lasso', 'Elastic-Net', 'Random Forest', 'Gradient Boosting', 'Multiple Linear', 'Ridge', 'Neural Network', 'KNN', 'SVR'),
                 levels = rev(c('Lasso', 'Elastic-Net', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'Multiple Linear', 'Ridge', 'Neural Network', 'KNN', 'SVR'))),
  R2 = c(87.81, 87.82, 87.82, 87.62, 87.70, 87.44, 87.19, 87.30, 86.66, 24.89),
  MAE = c(0.0750, 0.0750, 0.0750, 0.0752, 0.0754, 0.0761, 0.0768, 0.0764, 0.0772, 0.1535)
)
p1 <- ggplot(df, aes(x = Model, y = R2)) +
  geom_col(fill = '#275E9C', width = 0.65) +
  geom_text(aes(label = sprintf('%.2f%%', R2)), hjust = -0.15, size = 3.5, fontface = 'bold', color = '#1F2D3D') +
  coord_flip(ylim = c(0, 100)) +
  theme_minimal(base_size = 12) +
  labs(title = 'Ten Regression Models: Held-Out Test R-Squared (R2)',
       subtitle = 'All 10 models trained on ALL 20 input variables (998 test observations)',
       x = '', y = 'R-Squared (%)') +
  theme(plot.title = element_text(face = 'bold', color = '#1F2D3D'),
        axis.text = element_text(color = '#1F2D3D', face = 'bold'),
        panel.grid.minor = element_blank())
ggsave('DA2_DA3/results/figures/fig_regression_10_models_r2.png', p1, width = 8, height = 5, dpi = 150)

# Figure 2: Old Single Variable vs All 20 Variables
df_comp <- data.frame(
  Model = rep(c('Multiple Linear', 'Ridge', 'Lasso', 'KNN', 'Random Forest', 'Decision Tree'), each = 2),
  Version = rep(c('Old (Single Variable: ai_intensity_score)', 'New (All 20 Input Variables)'), 6),
  R2 = c(78.85, 87.44, 78.37, 87.19, 79.03, 87.82, 61.61, 86.66, 85.90, 87.62, 87.52, 87.81)
)
df_comp$Model <- factor(df_comp$Model, levels = c('KNN', 'Ridge', 'Multiple Linear', 'Lasso', 'Random Forest', 'Decision Tree'))
p2 <- ggplot(df_comp, aes(x = Model, y = R2, fill = Version)) +
  geom_col(position = position_dodge(0.75), width = 0.65) +
  geom_text(aes(label = sprintf('%.1f%%', R2)), position = position_dodge(0.75), hjust = -0.1, size = 3.2, fontface = 'bold') +
  scale_fill_manual(values = c('Old (Single Variable: ai_intensity_score)' = '#A0AEC0', 'New (All 20 Input Variables)' = '#008991')) +
  coord_flip(ylim = c(0, 100)) +
  theme_minimal(base_size = 12) +
  labs(title = 'Model Performance Improvement: All 20 Variables vs. Previous Single Variable',
       subtitle = 'Comparison against old baseline directly answering faculty feedback',
       x = '', y = 'Held-Out Test R2 (%)', fill = '') +
  theme(plot.title = element_text(face = 'bold', color = '#1F2D3D'),
        legend.position = 'bottom',
        axis.text = element_text(color = '#1F2D3D', face = 'bold'),
        panel.grid.minor = element_blank())
ggsave('DA2_DA3/results/figures/fig_regression_all_vs_old.png', p2, width = 8.5, height = 5, dpi = 150)
cat('Figures successfully generated.\n')
