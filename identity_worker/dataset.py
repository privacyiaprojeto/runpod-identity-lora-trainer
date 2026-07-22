from __future__ import annotations
import csv
from pathlib import Path
from .storage import download_private

def materialize_dataset(request, settings, work_dir: Path, s3) -> tuple[Path, Path]:
    dataset_root = work_dir / 'dataset'
    assets_root = dataset_root / 'assets'
    rows = []
    cache: dict[tuple[str,str,str], Path] = {}
    for sample in request.payload['dataset']['samples']:
        video_ref = sample['video_source']; image_ref = sample['reference_image_source']
        vkey=(video_ref['bucket'],video_ref['key'],sample['video_sha256'])
        ikey=(image_ref['bucket'],image_ref['key'],sample['reference_image_sha256'])
        if vkey not in cache:
            cache[vkey]=download_private(s3,*vkey[:2],assets_root/f"video_{len(cache):03d}.mp4",vkey[2])
        if ikey not in cache:
            cache[ikey]=download_private(s3,*ikey[:2],assets_root/f"image_{len(cache):03d}.jpg",ikey[2])
        video_rel=cache[vkey].relative_to(dataset_root).as_posix(); image_rel=cache[ikey].relative_to(dataset_root).as_posix()
        rows.append({'video':video_rel,'vace_video':video_rel,'vace_reference_image':image_rel,'prompt':sample['prompt']})
    metadata_path=dataset_root/'metadata.csv'
    dataset_root.mkdir(parents=True,exist_ok=True)
    with metadata_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['video','vace_video','vace_reference_image','prompt']); writer.writeheader(); writer.writerows(rows)
    return dataset_root, metadata_path
