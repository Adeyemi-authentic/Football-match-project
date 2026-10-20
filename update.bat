@echo off
echo Pushing changes to GitHub...
git add .
git commit -m "auto-update: %date% %time%"
git pull --rebase origin main
git push -u origin main
echo.
echo Done! Changes pushed to GitHub.
pause
