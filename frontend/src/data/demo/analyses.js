/**
 * Isolated presentation demo data for FaceSense frontend showcase.
 * Represents ONLY actual FaceSense capabilities:
 * - Face bounding boxes
 * - Emotion prediction (among 7 classes: angry, disgust, fear, happy, neutral, sad, surprise)
 * - Confidence score
 * - Class probabilities
 * (NO facial landmarks, eye contact, head pose, or action units)
 */

export const DEMO_SAMPLE_IMAGE = "https://lh3.googleusercontent.com/aida/AEtjO1UPufS0AHbuYZr7EK_XhZRXOv0eLx2_Q45WNEBYOhkgcnwWXmakBbmU-lC5q1hq9HYYR8oy5cATcD1J6dCwuq4QDuvP-FGNm5NDu4_EP2z6LVbOdc5EyA84vAvI5BokVoGKYNkk1I9KleGP1w2fdRuDqfP_Is0PFldviPmn1nZkCcAufY1OznXuevqEDUW5NqcXbezQ4T9ahBuOID1hgJOxY2Azlmp9haxU0Y9W9vSaUNR1JbCafCSgG7A";

export const DEMO_FACES = [
  {
    id: "01",
    label: "Subject 01",
    sublabel: "Primary Focus",
    // Percent coordinates [top, left, width, height]
    box: { top: 14, left: 19, width: 23, height: 35 },
    predicted_emotion: "happy",
    confidence: 0.974,
    probabilities: {
      happy: 0.974,
      neutral: 0.018,
      surprise: 0.006,
      sad: 0.001,
      fear: 0.0005,
      disgust: 0.0003,
      angry: 0.0002,
    },
  },
  {
    id: "02",
    label: "Subject 02",
    sublabel: "Secondary",
    box: { top: 10, left: 60, width: 21, height: 34 },
    predicted_emotion: "neutral",
    confidence: 0.731,
    probabilities: {
      neutral: 0.731,
      sad: 0.182,
      surprise: 0.045,
      happy: 0.021,
      fear: 0.012,
      angry: 0.006,
      disgust: 0.003,
    },
  },
];

export const DEMO_HISTORY_RECORDS = [
  {
    id: "rec_001",
    filename: "editorial_session_04.jpg",
    date: "2026-09-11 14:32",
    faces_count: 2,
    primary_emotion: "happy",
    confidence: 0.974,
    feedback_state: "correct",
    thumbnail: DEMO_SAMPLE_IMAGE,
  },
  {
    id: "rec_002",
    filename: "client_interview_batch_12.jpg",
    date: "2026-09-11 11:15",
    faces_count: 1,
    primary_emotion: "neutral",
    confidence: 0.884,
    feedback_state: "correct",
    thumbnail: DEMO_SAMPLE_IMAGE,
  },
  {
    id: "rec_003",
    filename: "studio_portrait_take_09.png",
    date: "2026-09-10 17:40",
    faces_count: 3,
    primary_emotion: "surprise",
    confidence: 0.912,
    feedback_state: "uncertain",
    thumbnail: DEMO_SAMPLE_IMAGE,
  },
  {
    id: "rec_004",
    filename: "user_test_reaction_03.jpg",
    date: "2026-09-10 13:05",
    faces_count: 1,
    primary_emotion: "sad",
    confidence: 0.652,
    feedback_state: "incorrect",
    thumbnail: DEMO_SAMPLE_IMAGE,
  },
  {
    id: "rec_005",
    filename: "team_discussion_cam2.jpg",
    date: "2026-09-09 09:22",
    faces_count: 4,
    primary_emotion: "happy",
    confidence: 0.941,
    feedback_state: "correct",
    thumbnail: DEMO_SAMPLE_IMAGE,
  },
];
