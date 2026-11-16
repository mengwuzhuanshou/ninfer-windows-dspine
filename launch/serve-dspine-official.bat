@echo off
rem DSpine K15 speculative decode - OFFICIAL target tower (default port 8080).
rem Prerequisite: fuse the official DFlash2 tower with the shipped DSpine draft suffix:
rem   python scripts\fuse_dspine_draft.py models\official-dflash2.ninfer ^
rem       draft\dspine-draft-official-nvfp4tab.ninfer models\official.dspine.ninfer
rem (use the -bf16tab suffix for BF16 codebook/tg/tsv tables instead of nvfp4).
rem Keep this file ASCII-only and CRLF.
title ninfer-serve-dspine-official - DSpine K15 official (port 8080)
cd /d %~dp0..

bin\ninfer-serve-dspine.exe models\official.dspine.ninfer ^
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
