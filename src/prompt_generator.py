from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_batch_prompt(batch: pd.DataFrame, output_path: Path, batch_id: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_batch_prompt(batch, batch_id), encoding="utf-8")
    return output_path


def build_batch_prompt(batch: pd.DataFrame, batch_id: str) -> str:
    indicator_lines = []
    for _, row in batch.iterrows():
        indicator_lines.append(
            "\n".join(
                [
                    f"- {row['indicator_id']} | {row.get('indicator_name', '')}",
                    f"  Pillar: {row.get('pillar', '')}",
                    f"  Framework: {row.get('framework', '')}",
                    f"  Definition: {row.get('definition', '')}",
                ]
            )
        )

    return f"""# ESG Disclosure Quality Scoring Prompt

Batch ID: {batch_id}

Bạn là chuyên gia ESG.

Nhiệm vụ: chấm tất cả các dòng trong file CSV đi kèm. Chỉ sử dụng evidence trong CSV. Không suy luận ngoài bằng chứng. Khi không chắc chắn, chọn điểm thấp hơn.

## Rubric

0 = No disclosure

1 = Qualitative disclosure

2 = Quantitative disclosure

3 = Quantitative disclosure with targets or outcomes

## Indicator Definitions In This Batch

{chr(10).join(indicator_lines)}

## Required Output

Trả về CSV duy nhất với đúng các cột sau:

```csv
company,year,indicator_id,score,confidence,reasoning
```

Quy tắc:

- `score` phải là một trong `0,1,2,3`.
- `confidence` nằm trong `[0,1]`.
- `reasoning` ngắn gọn, nêu evidence chính và lý do chọn điểm.
- Không thêm markdown ngoài CSV kết quả.
"""
