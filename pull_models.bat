@echo off
echo Pulling required Ollama models...
echo --------------------------------
echo 1. qwen2.5-coder:1.5b (Filter)
ollama pull qwen2.5-coder:1.5b

echo.
echo 2. llama3.2:3b (Mapper)
ollama pull llama3.2:3b

echo.
echo 3. deepseek-r1:1.5b (Synthesis)
ollama pull deepseek-r1:1.5b

echo.
echo All models pulled! You can now run the app.
pause
