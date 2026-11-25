// Is a chat message a request for a recurring task? Deliberately narrow: an
// ordinary question that mentions "every day" must be sent as a message.

const DAY = "(?:mon|tue|tues|wed|weds|thu|thur|thurs|fri|sat|sun)(?:day|sday|nesday|rsday|urday)?s?";
const UNIT = `(?:single\\s+)?(?:day|weekday|weekend|week|month|morning|afternoon|evening|night|hour|(?:\\d+|two|three|four|six|eight|twelve)\\s+hours?|${DAY})`;
const PHRASE = `(?:(?:every|each)\\s+${UNIT}|daily|weekly|hourly|nightly)`;
const TIME = "(?:at|by|around)\\s+(?:noon|midnight|\\d{1,2}(?::\\d{2})?\\s*(?:[ap]\\.?m\\.?)?)";

const STARTS_WITH_TIME = new RegExp(`^${PHRASE}\\b.{0,60}?\\b${TIME}`, "i");
const STARTS_WITH_COMMA = new RegExp(`^${PHRASE}\\s*,\\s*\\w+`, "i");
const ENDS_WITH_TIME = new RegExp(`\\s${PHRASE}\\s+${TIME}\\s*\\.?$`, "i");

export function routineRequest(message) {
  const text = message.trim();
  const command = /^\/routine(?:\s+([\s\S]*))?$/i.exec(text);
  if (command) return { text: (command[1] || "").trim(), explicit: true };
  if (text.endsWith("?")) return null;
  const match = STARTS_WITH_TIME.test(text) || STARTS_WITH_COMMA.test(text) || ENDS_WITH_TIME.test(text);
  return match ? { text, explicit: false } : null;
}
