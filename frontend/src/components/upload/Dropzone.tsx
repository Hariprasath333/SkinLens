import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, AlertCircle } from 'lucide-react';

interface DropzoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({ onFilesSelected, disabled = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files));
      // reset so same files can be re-selected if removed
      e.target.value = '';
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && fileInputRef.current?.click()}
      className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-all cursor-pointer ${
        disabled
          ? 'border-zinc-200 bg-zinc-50 opacity-60 cursor-not-allowed'
          : isDragOver
          ? 'border-primary-500 bg-primary-50/40 shadow-sm'
          : 'border-zinc-300 bg-white hover:border-zinc-400 hover:bg-zinc-50/50'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled}
      />

      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100 text-zinc-600">
        <UploadCloud className="h-6 w-6 text-zinc-600" />
      </div>

      <div className="mb-1 text-sm font-semibold text-zinc-900">
        Drop dermoscopic lesion images here, or <span className="text-primary-600 underline">browse</span>
      </div>

      <p className="text-xs text-zinc-500">
        Supports JPEG, PNG, WEBP (up to 15MB each, max 15 images per batch)
      </p>

      <div className="mt-4 flex items-center gap-4 text-[11px] text-zinc-400">
        <span className="flex items-center gap-1">
          <ImageIcon className="h-3.5 w-3.5" /> High resolution supported
        </span>
        <span>•</span>
        <span>Independent metadata per image</span>
      </div>
    </div>
  );
};
