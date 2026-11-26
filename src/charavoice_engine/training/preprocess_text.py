import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

from tqdm import tqdm

from charavoice_engine.constants import Languages
from charavoice_engine.nlp import clean_text

logger = logging.getLogger(__name__)


# Count lines for tqdm
def count_lines(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def write_error_log(error_log_path: Path, line: str, error: Exception):
    with error_log_path.open("a", encoding="utf-8") as error_log:
        error_log.write(f"{line.strip()}\n{error}\n\n")


def process_line(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    yomi_error: str,
):
    splitted_line = line.strip().split("|")
    if len(splitted_line) != 4:
        raise ValueError(f"Invalid line format: {line.strip()}")
    utt, spk, language, text = splitted_line
    language_enum = Languages(language)
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        language=language_enum,
        raise_yomi_error=(yomi_error != "use"),
    )
    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join([str(i) for i in tones]),
        " ".join([str(i) for i in word2ph]),
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Path | None,
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_speaker: int,
    max_val_total: int,
    # clean: bool,
    yomi_error: str,
    correct_path: bool,
    seed: int = 1234,
):
    assert yomi_error in ["raise", "skip", "use"]
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path.with_name(transcription_path.name + ".cleaned")
    train_tmp_path = train_path.with_name(train_path.name + ".tmp")
    val_tmp_path = val_path.with_name(val_path.name + ".tmp")
    for stale_path in (train_tmp_path, val_tmp_path):
        if stale_path.exists():
            stale_path.unlink()

    error_log_path = transcription_path.parent / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    total_lines = count_lines(transcription_path)

    # transcription_path から 1行ずつ読み込んで文章処理して cleaned_path に書き込む
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(trans_file, file=sys.stdout, total=total_lines, dynamic_ncols=True):
            try:
                processed_line = process_line(
                    line,
                    transcription_path,
                    correct_path,
                    yomi_error,
                )
                out_file.write(processed_line)
            except Exception as e:
                logger.error("An error occurred at line:\n%s\n%s", line.strip(), e)
                write_error_log(error_log_path, line, e)
                error_count += 1

    transcription_path = cleaned_path

    # 各話者ごとのlineの辞書
    spk_utt_map: dict[str, list[str]] = defaultdict(list)

    # 話者からIDへの写像
    spk_id_map: dict[str, int] = {}

    # 話者ID
    current_sid: int = 0

    # 音源ファイルのチェックや、spk_id_mapの作成
    with transcription_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        count_same = 0
        count_not_found = 0
        for line in f.readlines():
            utt, spk = line.strip().split("|")[:2]
            if utt in audio_paths:
                logger.warning(f"Same audio file appears multiple times: {utt}")
                count_same += 1
                continue
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                count_not_found += 1
                continue
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)

            # 新しい話者が出てきたら話者IDを割り当て、current_sidを1増やす
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
        if count_same > 0 or count_not_found > 0:
            raise ValueError(
                f"Invalid transcription file: repeated audios={count_same}, audio not found={count_not_found}"
            )

    train_list: list[str] = []
    val_list: list[str] = []

    # 各話者ごとに発話リストを処理
    rng = Random(seed)
    for _spk, utts in spk_utt_map.items():
        if val_per_speaker == 0:
            train_list.extend(utts)
            continue
        if len(utts) <= val_per_speaker:
            val_indices = set(range(len(utts)))
        else:
            val_indices = set(rng.sample(range(len(utts)), val_per_speaker))
        # 元の順序を保ちながらリストを分割
        for index, utt in enumerate(utts):
            if index in val_indices:
                val_list.append(utt)
            else:
                train_list.append(utt)

    # バリデーションリストのサイズ調整
    if len(val_list) > max_val_total:
        extra_val = val_list[max_val_total:]
        val_list = val_list[:max_val_total]
        # 余剰のバリデーション発話をトレーニングリストに追加 (元の順序を保持)
        train_list.extend(extra_val)

    if not train_list:
        raise ValueError(
            "Training list is empty after validation split. Increase dataset size or reduce --val-per-speaker."
        )

    with train_tmp_path.open("w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with val_tmp_path.open("w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    with config_path.open("r", encoding="utf-8") as f:
        json_config = json.load(f)

    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)

    if error_count > 0 and yomi_error != "skip":
        for stale_path in (train_tmp_path, val_tmp_path):
            if stale_path.exists():
                stale_path.unlink()
        # yomi_error == "use" では yomi_error=False で処理しているため、
        # ここに来るのは読み補正以外の例外が混ざった場合。
        logger.error(f"An error occurred in {error_count} lines. Please check {error_log_path} for details.")
        raise Exception(
            f"An error occurred in {error_count} lines. Please check `Data/you_model_name/text_error.log` file for details."
        )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)

    train_tmp_path.replace(train_path)
    val_tmp_path.replace(val_path)

    if error_count > 0:
        logger.warning(
            f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
        )
    else:
        logger.info("Training set and validation set generation from texts is complete!")


if __name__ == "__main__":
    from charavoice_engine.nlp.japanese import pyopenjtalk_worker

    pyopenjtalk_worker.initialize_worker()

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcription-path", required=True)
    parser.add_argument("--cleaned-path")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--config-path", required=True)

    parser.add_argument(
        "--val-per-speaker",
        type=int,
        default=2,
        help="Number of validation utterances to hold out per speaker.",
    )
    parser.add_argument("--max-val-total", type=int, default=100)
    parser.add_argument("--yomi_error", choices=["raise", "skip", "use"], default="raise")
    parser.add_argument("--correct_path", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    transcription_path = Path(args.transcription_path)
    cleaned_path = Path(args.cleaned_path) if args.cleaned_path else None
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    config_path = Path(args.config_path)
    val_per_speaker = int(args.val_per_speaker)
    max_val_total = int(args.max_val_total)
    yomi_error: str = args.yomi_error
    correct_path: bool = args.correct_path

    preprocess(
        transcription_path=transcription_path,
        cleaned_path=cleaned_path,
        train_path=train_path,
        val_path=val_path,
        config_path=config_path,
        val_per_speaker=val_per_speaker,
        max_val_total=max_val_total,
        yomi_error=yomi_error,
        correct_path=correct_path,
        seed=args.seed,
    )
