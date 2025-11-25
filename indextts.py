import os
from pathlib import Path
from indextts.infer_v2 import IndexTTS2


def generate_output_path(
    spk_audio_prompt, emo_audio_prompt, text, base_output_dir="outputs"
):
    """
    构建输出路径为src名称/text截取_refer
    """
    spk_name = Path(spk_audio_prompt).stem
    emo_name = Path(emo_audio_prompt).stem
    clean_text = text.replace("\n", " ").replace("\r", " ").strip()
    text_prefix = clean_text[:8] if clean_text else "empty"
    output_filename = f"{emo_name}_{text_prefix}.wav"
    output_path = os.path.join(base_output_dir, spk_name, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return output_path


# 初始化TTS模型
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False,
)

# 用户可修改的变量
text = """

"""
spk_audio_prompt = "src/yw.WAV"  # 修改这里以使用不同的说话人音频
emo_audio_prompt = "refer/mixed moan.WAV"  # 修改这里以使用不同的情感参考音频

# 生成输出路径
output_path = generate_output_path(
    spk_audio_prompt=spk_audio_prompt, emo_audio_prompt=emo_audio_prompt, text=text
)

# 执行推理
tts.infer(
    spk_audio_prompt=spk_audio_prompt,
    text=text,
    output_path=output_path,
    emo_audio_prompt=emo_audio_prompt,
    emo_alpha=0.75,
    verbose=True,
)
