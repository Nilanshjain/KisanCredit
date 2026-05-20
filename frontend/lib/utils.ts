import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Framer-motion v12 redefines onDrag/onAnimationStart/onAnimationEnd/onAnimationIteration
 * with different signatures than React's DOM types. Spreading native HTML props onto a
 * motion.X component fails TS strict-mode compilation. Strip those keys with this helper.
 */
export type MotionSafe<T> = Omit<
  T,
  | 'onDrag'
  | 'onDragStart'
  | 'onDragEnd'
  | 'onAnimationStart'
  | 'onAnimationEnd'
  | 'onAnimationIteration'
>;
