@echo off
rem DSpine K15 speculative decode - CODER390 author fine-tune tower (default port 8080).
rem Prerequisite: fuse the Coder390 tokfix tower with the shipped DSpine draft suffix:
rem   python scripts\fuse_dspine_draft.py models\coder390-tokfix.ninfer ^
rem       draft\dspine-draft-coder390-nvfp4tab.ninfer models\coder390.dspine.ninfer
rem The Coder390 suffix carries TG/TSV tables computed for that tower's head; do not mix
rem official and Coder390 suffixes. Keep this file ASCII-only and CRLF.
title ninfer-serve-dspine-coder390 - DSpine K15 author (port 8080)
cd /d %~dp0..

bin\ninfer-serve-dspine.exe models\coder390.dspine.ninfer ^
  --model-id t ^
  --max-context 32768 ^
  --default-max-tokens 512 ^
  --max-concurrency 1 ^
  --max-pending-requests 4 ^
  --kv-dtype nvfp4 ^
  --host-kv-mib 512 ^
  --no-thinking ^
  --spec dspine --draft-tokens 15 ^
  --cors ^
  --host 127.0.0.1 ^
  --port 8080

pause
