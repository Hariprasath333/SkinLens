# Screenshots (phone, 390 x 844)

Taken by `tools/mobile_check.py` in Microsoft Edge with phone emulation (Android user agent, touch, 390 x 844), against a DEMO database (`seed_demo.py`). Run it again after any change to refresh them.

| File | Shows |
|---|---|
| `01_start_screen.png` | Start screen: choose doctor or patient mode, "Install SkinLens" button |
| `02_patient_upload.png` | Patient mode: Take photo / Choose from gallery, preview (a 4032×3024 photo made smaller to 1024×768 on the phone), tips |
| `03_patient_result.png`, `03b_patient_result_full.png` | Patient result: advice, swipeable photo / heat-map tabs, plain-language result, confidence level, Download summary |
| `04_doctor_dashboard.png` | Doctor dashboard with the bottom navigation bar |
| `05_doctor_result.png`, `05b_doctor_result_full.png` | Doctor result: banner, Image / Grad-CAM / LIME tabs, explanation, top 3, feedback |
| `06_doctor_report_pdf.png` | First page of the doctor's PDF report |
| `06b_patient_summary_pdf.png` | The one-page summary a patient can show their doctor |
| `mobile_check.json` | Check results: horizontal scrolling at 390 and 360 px (13 pages), tap targets, PWA installability, service-worker cache contents |

Images marked DEMO and the "DEMO Patient" names come from demo data, not real patients. The full-page (`b`) screenshots show the bottom bar in the middle of the page; this is how full-page captures render fixed elements, not a layout problem.

## New features (v1.5), taken by `tools/feature_screens.py`

`tools/feature_screens.py` starts its own server with a throw-away database, creates a DEMO admin and a DEMO patient, and analyses a HAM10000 **test** image (a public dataset image, not a real patient). Desktop shots are 1366×768, phone shots 390×844, both at scale 2.

| File | Shows |
|---|---|
| `07_new_logo_login_desktop.png`, `07b_new_logo_start_phone.png` | New logo on the login page and the start screen, with the ABCDE link |
| `09_patient_about_you_always_open_phone.png` | Patient mode: "About you (optional)" always open |
| `10_doctor_quality_warning_blurred_desktop.png`, `11_patient_quality_warning_blurred_phone.png`, `11_patient_quality_warning_dark_phone.png` | Photo quality check: blurred and dark photo, with "Retake photo" / "Use anyway" |
| `12a_patient_result_top_phone.png`, `12_patient_result_probability_bars_phone.png` | Patient result: advice, and all 7 probabilities as coloured bars with the calm note |
| `13_patient_why_this_result_phone.png`, `14_patient_full_details_phone.png`, `14b_patient_result_full_page_phone.png` | "Why the app gave this result": measured reasons, heat-map and LIME side by side, "Show full details" opened |
| `15_doctor_result_desktop.png`, `15b_doctor_result_explanation_desktop.png`, `15c_doctor_result_full_desktop.png`, `16_doctor_result_bars_phone.png` | Doctor result: quality check, measurement table, clinical summary, all 7 probabilities |
| `17_disease_page_melanoma_phone.png`, `17b_disease_list_phone.png`, `18_abcde_page_phone.png` | Disease information pages and the ABCDE self-check |
| `19_doctor_report_pdf_p1.png`, `19_doctor_report_pdf_p2.png`, `20_patient_summary_pdf_p1.png`, `20_patient_summary_pdf_p2.png` | PDF reports with the logo, the probability table and the explanation |
| `22_about_page_desktop.png` | About / version page: model SHA-256 and validation status |

## Validation Lab (TRL 4), taken by `tools/feature_screens.py --validation`

| File | Shows |
|---|---|
| `23_validation_lab_desktop.png`, `23b_validation_lab_full_desktop.png` | Validation Lab page: run buttons, the frozen acceptance criteria, results with tables and charts |
| `23c_validation_progress_desktop.png` | Final run `2026-11-15_182522`: T1–T7 all PASS |
| `22_about_page_desktop.png` | About / version page: "Validated: TRL 4 (laboratory) – 2026-11-16", model SHA-256 |
| `24_trl4_report_p1.png` … `24_trl4_report_p12.png` | All 12 pages of the TRL-4 Laboratory Validation Report |

`../yukti_photo.jpg` (1600×1000, 267 KB) is a collage of four of these screenshots, made by `tools/yukti_photo.py`.
