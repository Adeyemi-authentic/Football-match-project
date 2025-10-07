@echo off
echo Pushing changes to GitHub...
git add .
git commit -m "auto-update: %date% %time%"
git push -u origin amor
echo.
echo Done! Changes pushed to GitHub.
pause