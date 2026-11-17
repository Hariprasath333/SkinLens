export type Sex = 'female' | 'male' | 'unknown';

export const ANATOMICAL_SITES = [
  'abdomen',
  'back',
  'chest',
  'ear',
  'face',
  'foot',
  'genital',
  'hand',
  'lower extremity',
  'neck',
  'scalp',
  'trunk',
  'upper extremity',
  'acral',
  'oral/genital',
  'unknown'
] as const;

export type AnatomicalSite = typeof ANATOMICAL_SITES[number];

export const CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] as const;
export type ClassKey = typeof CLASS_NAMES[number];

export interface ClassMetadata {
  key: ClassKey;
  code: string;
  name: string;
  category: 'Benign' | 'Malignant' | 'Precancerous / In-situ';
  description: string;
  clinicalSignificance: string;
}

export const CLASS_DETAILS: Record<ClassKey, ClassMetadata> = {
  akiec: {
    key: 'akiec',
    code: 'AKIEC',
    name: 'Actinic Keratosis / Intraepithelial Carcinoma',
    category: 'Precancerous / In-situ',
    description: 'Precancerous squamous cell proliferations including Actinic Keratoses and Bowen\'s Disease.',
    clinicalSignificance: 'Commonly caused by cumulative ultraviolet radiation exposure; may progress to invasive squamous cell carcinoma if left untreated.'
  },
  bcc: {
    key: 'bcc',
    code: 'BCC',
    name: 'Basal Cell Carcinoma',
    category: 'Malignant',
    description: 'Common form of non-melanoma skin cancer arising from epidermal basal cells.',
    clinicalSignificance: 'Slow-growing and rarely metastasizes, but can cause substantial local tissue destruction without prompt intervention.'
  },
  bkl: {
    key: 'bkl',
    code: 'BKL',
    name: 'Benign Keratosis-like Lesions',
    category: 'Benign',
    description: 'Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses (LPLK).',
    clinicalSignificance: 'Non-cancerous epithelial proliferations frequently encountered in older adults; requires no surgical treatment unless irritated.'
  },
  df: {
    key: 'df',
    code: 'DF',
    name: 'Dermatofibroma',
    category: 'Benign',
    description: 'Benign fibrohistiocytic cutaneous nodule located in the reticular dermis.',
    clinicalSignificance: 'Harmless dermal lesion often secondary to minor trauma or insect stings, showing characteristic "dimple sign" on lateral compression.'
  },
  mel: {
    key: 'mel',
    code: 'MEL',
    name: 'Melanoma',
    category: 'Malignant',
    description: 'Malignant neoplasm of melanocytic origin with marked propensity for early invasion and metastasis.',
    clinicalSignificance: 'Most dangerous form of skin cancer. Early surgical excision is the primary determinant of long-term patient survival.'
  },
  nv: {
    key: 'nv',
    code: 'NV',
    name: 'Melanocytic Nevi',
    category: 'Benign',
    description: 'Common benign proliferation of melanocytes (moles), forming nested architectures.',
    clinicalSignificance: 'Extremely prevalent benign pigmented lesions. Symmetrical, stable over time, and benign unless undergoing dysplastic transformation.'
  },
  vasc: {
    key: 'vasc',
    code: 'VASC',
    name: 'Vascular Lesions',
    category: 'Benign',
    description: 'Vascular abnormalities including cherry angiomas, pyogenic granulomas, and angiokeratomas.',
    clinicalSignificance: 'Benign proliferations or malformations of cutaneous blood vessels; clinically benign with risk of bleeding.'
  }
};

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  device: string;
  model_name: string;
  version: string;
  temperature: number;
  num_classes: number;
}

export interface ClassItem {
  index: number;
  key: string;
  name: string;
  description: string;
}

export interface ClassesResponse {
  total: number;
  classes: ClassItem[];
}

export interface GradCamResponse {
  available: boolean;
  target_layer?: string;
  target_class?: string;
  scope?: string;
  original_image?: string; // base64 data URL
  heatmap?: string;        // base64 data URL
  overlay?: string;        // base64 data URL
  disclaimer?: string;
  error?: string;
}

export interface PredictResponse {
  item_id: string;
  status: 'completed' | 'failed';
  predicted_class?: string;
  predicted_class_name?: string;
  raw_confidence?: number;
  calibrated_confidence?: number;
  raw_probabilities?: Record<string, number>;
  calibrated_probabilities?: Record<string, number>;
  metadata_used: {
    age?: number | null;
    sex?: string | null;
    localization?: string | null;
  };
  gradcam?: GradCamResponse;
  error?: string;
}

export interface BatchItemResult {
  id: string;
  status: 'completed' | 'failed';
  predicted_class?: string;
  predicted_class_name?: string;
  raw_confidence?: number;
  calibrated_confidence?: number;
  raw_probabilities?: Record<string, number>;
  calibrated_probabilities?: Record<string, number>;
  metadata_used: {
    age?: number | null;
    sex?: string | null;
    localization?: string | null;
  };
  gradcam?: GradCamResponse;
  error?: string;
}

export interface BatchResponse {
  total_items: number;
  successful_items: number;
  failed_items: number;
  results: BatchItemResult[];
}

export interface ExplainResponse {
  predicted_class: string;
  target_class: string;
  target_class_name: string;
  target_class_cal_prob: number;
  target_class_raw_prob: number;
  original_image: string;
  heatmap: string;
  overlay: string;
  target_layer: string;
  scope: string;
  disclaimer: string;
}

export interface QueueItem {
  id: string;
  file: File;
  previewUrl: string;
  age: string; // string input in form
  sex: Sex;
  localization: string;
  status: 'ready' | 'processing' | 'completed' | 'failed';
  result?: BatchItemResult;
  error?: string;
}
