if not contains "$HOME/AI_Main_Repo/CV_parsing_main/eraparse/artifacts/tools/uv-arm" $PATH
    # Prepending path in case a system-installed binary needs to be overridden
    set -x PATH "$HOME/AI_Main_Repo/CV_parsing_main/eraparse/artifacts/tools/uv-arm" $PATH
end
