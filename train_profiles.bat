@echo off
setlocal
echo ============================================================
echo  BATCH TRAINING - 4 profils de modeles
echo ============================================================
echo.

REM ========== PROFIL 1 : curriculum ==========
echo [1/4] Profil CURRICULUM (2000 episodes, 3 bots, progression graduelle)
python -u train_dqn.py ^
  --output-dir outputs_curriculum ^
  --episodes 2000 ^
  --curriculum ^
  --curriculum-stage-1 600 ^
  --curriculum-stage-2 1200 ^
  --n-bots 3 ^
  --lr 3e-4 ^
  --gamma 0.99 ^
  --epsilon-decay 0.998 ^
  --win-bonus 35.0 ^
  --death-penalty -20.0 ^
  --survival-reward 0.03 ^
  --seed 42
echo [1/4] Curriculum termine.
echo.

REM ========== PROFIL 2 : agressif ==========
echo [2/4] Profil AGRESSIF (2000 episodes, fort bonus kill/win)
python -u train_dqn.py ^
  --output-dir outputs_agressif ^
  --episodes 2000 ^
  --n-bots 3 ^
  --lr 4e-4 ^
  --gamma 0.97 ^
  --epsilon-decay 0.997 ^
  --epsilon-end 0.08 ^
  --win-bonus 60.0 ^
  --elimination-bonus 5.0 ^
  --death-penalty -10.0 ^
  --survival-reward 0.01 ^
  --loop-penalty -3.0 ^
  --seed 123
echo [2/4] Agressif termine.
echo.

REM ========== PROFIL 3 : conservateur ==========
echo [3/4] Profil CONSERVATEUR (2000 episodes, survie prioritaire)
python -u train_dqn.py ^
  --output-dir outputs_conservateur ^
  --episodes 2000 ^
  --n-bots 3 ^
  --lr 2e-4 ^
  --gamma 0.995 ^
  --epsilon-decay 0.9985 ^
  --epsilon-end 0.03 ^
  --win-bonus 20.0 ^
  --death-penalty -30.0 ^
  --survival-reward 0.08 ^
  --elimination-bonus 0.5 ^
  --loop-penalty -2.0 ^
  --seed 7
echo [3/4] Conservateur termine.
echo.

REM ========== PROFIL 4 : explorateur ==========
echo [4/4] Profil EXPLORATEUR (2500 episodes, reseau large, exploration lente)
python -u train_dqn.py ^
  --output-dir outputs_explorateur ^
  --episodes 2500 ^
  --curriculum ^
  --n-bots 3 ^
  --hidden-dim 512 ^
  --lr 2e-4 ^
  --gamma 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-end 0.02 ^
  --epsilon-decay 0.9992 ^
  --batch-size 256 ^
  --buffer-capacity 200000 ^
  --win-bonus 35.0 ^
  --death-penalty -20.0 ^
  --seed 99
echo [4/4] Explorateur termine.
echo.

echo ============================================================
echo  Tous les profils sont entraines !
echo  Modeles disponibles dans :
echo    outputs_curriculum\models\best_eval_model.pt
echo    outputs_agressif\models\best_eval_model.pt
echo    outputs_conservateur\models\best_eval_model.pt
echo    outputs_explorateur\models\best_eval_model.pt
echo ============================================================
pause
