import { Upload } from 'lucide-react';
import { useRef } from 'react';

export default function ImageUpload({ onDetect, uploadedImage }) {
  const fileRef = useRef(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      setTimeout(() => {
        onDetect(['tomato', 'onion', 'garlic', 'pasta'], rd.result);
      }, 1500);
    };
    rd.readAsDataURL(file);
  };

  return (
    <>
      <label
        className="flex flex-col items-center justify-center w-full h-64 border-dashed border-2 rounded-xl cursor-pointer bg-gray-50 hover:border-orange-500"
        onClick={() => fileRef.current.click()}
      >
        {uploadedImage ? (
          <img src={uploadedImage} alt="Uploaded food" className="w-full h-full object-cover rounded-xl" />
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-400 mb-3" />
            <p className="text-gray-600 font-medium">Tap to upload or take a photo</p>
          </>
        )}
      </label>
      <input ref={fileRef} type="file" className="hidden" accept="image/*" onChange={handleUpload} />
    </>
  );
}
