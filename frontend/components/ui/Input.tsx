'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { AlertCircle } from 'lucide-react'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helperText?: string
  required?: boolean
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
  fullWidth?: boolean
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      type = 'text',
      label,
      error,
      helperText,
      required = false,
      icon,
      iconPosition = 'left',
      fullWidth = true,
      ...props
    },
    ref
  ) => {
    return (
      <div className={cn('space-y-2', fullWidth && 'w-full')}>
        {label && (
          <label className={cn('label', required && 'label-required')}>
            {label}
          </label>
        )}

        <div className="relative">
          {icon && iconPosition === 'left' && (
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-stone-400">
              {icon}
            </div>
          )}

          <motion.input
            ref={ref}
            type={type}
            className={cn(
              'input',
              error && 'input-error',
              icon && iconPosition === 'left' && 'pl-12',
              icon && iconPosition === 'right' && 'pr-12',
              className
            )}
            style={{
              color: '#1C1917',
              backgroundColor: '#FFFFFF',
              WebkitTextFillColor: '#1C1917',
              caretColor: '#1C1917',
              ...props.style
            }}
            autoComplete={props.autoComplete || 'off'}
            whileFocus={{ scale: 1.01 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
            {...props}
          />

          {icon && iconPosition === 'right' && (
            <div className="absolute right-4 top-1/2 -translate-y-1/2 text-stone-400">
              {icon}
            </div>
          )}

          {error && (
            <div className="absolute right-4 top-1/2 -translate-y-1/2 text-clay-500">
              <AlertCircle className="w-5 h-5" />
            </div>
          )}
        </div>

        {error && (
          <motion.p
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-clay-600 flex items-center gap-1"
          >
            {error}
          </motion.p>
        )}

        {helperText && !error && (
          <p className="text-sm text-stone-500">{helperText}</p>
        )}
      </div>
    )
  }
)

Input.displayName = 'Input'

export default Input
