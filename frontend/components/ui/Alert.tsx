'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { cn, type MotionSafe } from '@/lib/utils'
import { CheckCircle2, AlertCircle, AlertTriangle, Info, X } from 'lucide-react'

export interface AlertProps extends MotionSafe<React.HTMLAttributes<HTMLDivElement>> {
  variant?: 'success' | 'warning' | 'error' | 'info'
  title?: string
  /** Convenience prop — rendered as the alert body when `children` is omitted. */
  message?: React.ReactNode
  onClose?: () => void
  icon?: React.ReactNode
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ className, variant = 'info', title, message, onClose, icon, children, ...props }, ref) => {
    const baseClasses = 'p-4 rounded-xl border flex items-start gap-3'

    const variants = {
      success: 'bg-field-50 border-field-200 text-field-800',
      warning: 'bg-harvest-50 border-harvest-200 text-harvest-800',
      error: 'bg-clay-50 border-clay-200 text-clay-800',
      info: 'bg-stone-100 border-stone-200 text-stone-800',
    }

    const icons = {
      success: <CheckCircle2 className="w-5 h-5 text-field-600 flex-shrink-0" />,
      warning: <AlertTriangle className="w-5 h-5 text-harvest-600 flex-shrink-0" />,
      error: <AlertCircle className="w-5 h-5 text-clay-600 flex-shrink-0" />,
      info: <Info className="w-5 h-5 text-stone-600 flex-shrink-0" />,
    }

    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={cn(baseClasses, variants[variant], className)}
        {...props}
      >
        <div className="flex-shrink-0">{icon || icons[variant]}</div>

        <div className="flex-1 min-w-0">
          {title && <div className="font-semibold mb-1">{title}</div>}
          <div className="text-sm">{children ?? message}</div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="flex-shrink-0 text-stone-400 hover:text-stone-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </motion.div>
    )
  }
)

Alert.displayName = 'Alert'

export default Alert
